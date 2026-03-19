"""
企业资料智能查询 Agent Skill (完善版)
基于 RAG 技术，支持：混合检索、重排序、多级缓存、查询改写
"""
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
import hashlib
import logging
import re
from pathlib import Path
from collections import defaultdict
from threading import Lock
from functools import wraps
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== 工具函数 ==============
def synchronized(func):
    """线程同步装饰器"""
    func.__lock__ = Lock()
    def wrapper(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)
    return wrapper


def timed_cache(ttl: int = 300):
    """带TTL的缓存装饰器"""
    cache = {}
    lock = Lock()

    def decorator(func):
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            with lock:
                if key in cache:
                    result, timestamp = cache[key]
                    if time.time() - timestamp < ttl:
                        return result

            result = func(*args, **kwargs)
            with lock:
                cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator


# ============== 数据结构 ==============
@dataclass
class UserQuery:
    """用户查询"""
    user_id: str
    query: str
    document_type: List[str]
    language: str = "zh"


@dataclass
class DocumentChunk:
    """文档片段"""
    document_id: str
    title: str
    content: str
    section: str
    page: int
    chunk_id: int
    doc_type: str = ""
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """检索结果"""
    document_id: str
    title: str
    content: str
    section: str
    page: int
    similarity: float
    rerank_score: float = 0.0
    doc_type: str = ""


@dataclass
class QueryResult:
    """查询结果"""
    status: str
    answer: Optional[str]
    citations: List[Dict[str, Any]]
    related_docs: List[Dict[str, Any]]
    message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class QueryRewriteResult:
    """查询改写结果"""
    original: str
    expanded: str
    keywords: List[str]
    filters: Dict


# ============== 查询处理 ==============
class QueryRewriter:
    """查询改写器 - 扩展查询意图"""

    def __init__(self):
        # 同义词词典
        self.synonyms = {
            "故障": ["异常", "问题", "损坏", "失效", "停机"],
            "维护": ["保养", "检修", "维修", "巡检"],
            "操作": ["使用", "运行", "控制", "启动"],
            "规程": ["流程", "步骤", "方法", "规范", "标准"],
            "应急": ["紧急", "突发", "预案", "响应"],
            "安全": ["防护", "保护", "风险", "隐患"],
            "参数": ["指标", "规格", "数值", "阈值"],
            "设备": ["机器", "装置", "系统", "设施"]
        }

        # 停用词
        self.stopwords = {"的", "了", "是", "在", "和", "与", "或", "等", "请问", "如何", "怎么", "什么"}

    def rewrite(self, query: str) -> QueryRewriteResult:
        """
        改写查询

        Returns:
            QueryRewriteResult
        """
        # 分词（简单实现）
        words = self._tokenize(query)

        # 提取关键词
        keywords = [w for w in words if w not in self.stopwords]

        # 扩展查询
        expanded_terms = set(keywords)
        for kw in keywords:
            if kw in self.synonyms:
                expanded_terms.update(self.synonyms[kw])

        expanded_query = " ".join(expanded_terms)

        # 提取过滤条件
        filters = self._extract_filters(query)

        return QueryRewriteResult(
            original=query,
            expanded=expanded_query,
            keywords=list(keywords),
            filters=filters
        )

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 移除标点
        text = re.sub(r'[，。！？、；：""''【】（）]', ' ', text)
        # 按空格分词
        words = text.split()
        return words

    def _extract_filters(self, query: str) -> Dict:
        """提取过滤条件"""
        filters = {}

        # 文档类型
        doc_types = []
        if "应急" in query or "预案" in query:
            doc_types.append("emergency_plan")
        if "操作" in query or "手册" in query:
            doc_types.append("operation_manual")
        if "技术" in query or "规格" in query:
            doc_types.append("technical_spec")

        if doc_types:
            filters["document_type"] = doc_types

        return filters


# ============== 文档编码器 ==============
class DocumentEncoder:
    """文档编码器 - 支持多模型"""

    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        device: str = "cuda",
        max_seq_length: int = 256
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Encoder initialized: {model_name}, dim={self.embedding_dim}")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """批量编码"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2归一化
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """单条编码"""
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding


# ============== 重排序模型 ==============
class Reranker:
    """重排序器 - 使用 Cross-Encoder 提升排序精度"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, max_length=512, device=self.device)
        logger.info(f"Reranker initialized: {model_name}")

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        重排序

        Args:
            query: 查询
            candidates: 候选结果
            top_k: 返回数量

        Returns:
            重排序后的结果
        """
        if not candidates:
            return []

        # 构建查询-文档对
        pairs = [(query, c.content) for c in candidates]

        # 评分
        scores = self.model.predict(pairs)

        # 按分数排序
        sorted_indices = np.argsort(scores)[::-1]

        # 更新分数并返回
        results = []
        for idx in sorted_indices[:top_k]:
            candidate = candidates[idx]
            candidate.rerank_score = float(scores[idx])
            results.append(candidate)

        return results


# ============== 向量检索 ==============
class HybridRetriever:
    """混合检索器 - 结合向量和关键词检索"""

    def __init__(
        self,
        encoder: DocumentEncoder,
        index_type: str = "IVF",
        use_bm25: bool = True
    ):
        self.encoder = encoder
        self.index_type = index_type
        self.use_bm25 = use_bm25

        self.index = None
        self.documents: List[DocumentChunk] = []
        self.bm25_scores: Dict[int, Dict[str, float]] = {} if use_bm25 else None

        self._init_index()

    def _init_index(self):
        """初始化索引"""
        if self.index_type == "IVF":
            nlist = 100
            quantizer = faiss.IndexFlatL2(self.encoder.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.encoder.embedding_dim, nlist)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.encoder.embedding_dim, 32)
        else:
            self.index = faiss.IndexFlatL2(self.encoder.embedding_dim)

    def build_index(self, documents: List[DocumentChunk], embeddings: np.ndarray):
        """构建索引"""
        self.documents = documents

        # 向量索引
        embeddings = embeddings.astype('float32')
        if self.index_type == "IVF":
            self.index.train(embeddings)
        self.index.add(embeddings)

        # BM25 索引
        if self.use_bm25:
            self._build_bm25_index()

        logger.info(f"Indexed {len(documents)} documents")

    def _build_bm25_index(self):
        """构建 BM25 索引"""
        # 简单 BM25 实现
        for i, doc in enumerate(self.documents):
            # 简单分词
            words = doc.content.lower().split()
            self.bm25_scores[i] = defaultdict(float)
            for word in words:
                self.bm25_scores[i][word] += 1

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        alpha: float = 0.7
    ) -> List[SearchResult]:
        """
        混合检索

        Args:
            query_embedding: 查询向量
            query_text: 查询文本
            top_k: 返回数量
            filters: 过滤条件
            alpha: 向量权重 (1-alpha为BM25权重)

        Returns:
            检索结果
        """
        # 向量检索
        query_vec = query_embedding.astype('float32').reshape(1, -1)
        vector_scores, vector_indices = self.index.search(query_vec, len(self.documents))

        # BM25 检索
        bm25_scores = self._bm25_search(query_text)

        # 融合分数
        combined_scores = {}
        for i, (score, idx) in enumerate(zip(vector_scores[0], vector_indices[0])):
            if idx == -1:
                continue

            # 向量分数 (L2距离转相似度)
            vec_sim = 1 / (1 + score)

            # BM25 分数
            bm25_sim = bm25_scores.get(idx, 0.0)

            # 加权融合
            combined = alpha * vec_sim + (1 - alpha) * bm25_sim

            # 应用过滤
            if filters and filters.get("document_type"):
                doc = self.documents[idx]
                if doc.doc_type not in filters["document_type"]:
                    continue

            combined_scores[idx] = combined

        # 排序
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建结果
        results = []
        for idx, score in sorted_indices[:top_k]:
            doc = self.documents[idx]
            results.append(SearchResult(
                document_id=doc.document_id,
                title=doc.title,
                content=doc.content,
                section=doc.section,
                page=doc.page,
                similarity=score,
                doc_type=doc.doc_type
            ))

        return results

    def _bm25_search(self, query: str) -> Dict[int, float]:
        """BM25 检索"""
        if not self.use_bm25:
            return {}

        # 分词
        query_words = query.lower().split()
        scores = {}

        # 计算 IDF (简化)
        N = len(self.documents)
        doc_freqs = defaultdict(int)
        for doc_scores in self.bm25_scores.values():
            for word in doc_scores:
                if word in query_words:
                    doc_freqs[word] += 1

        for i, doc_scores in self.bm25_scores.items():
            score = 0.0
            for word in query_words:
                if word in doc_scores:
                    tf = doc_scores[word]
                    idf = np.log((N - doc_freqs[word] + 0.5) / (doc_freqs[word] + 0.5) + 1)
                    score += tf * idf / (tf + 1)  # BM25 公式简化

            if score > 0:
                scores[i] = score

        return scores


# ============== 知识库 ==============
class DocumentKnowledgeBase:
    """文档知识库 - 支持增量更新"""

    def __init__(self, encoder: DocumentEncoder, reranker: Optional[Reranker] = None):
        self.encoder = encoder
        self.reranker = reranker
        self.retriever: Optional[HybridRetriever] = None

        self.documents: Dict[str, Dict] = {}  # document_id -> metadata
        self.chunks: List[DocumentChunk] = []

    def load_documents(
        self,
        documents: List[Dict],
        chunk_size: int = 500,
        overlap: int = 50
    ):
        """加载并分块文档"""
        self.chunks = []
        all_texts = []

        for doc in documents:
            doc_id = doc["id"]
            title = doc["title"]
            content = doc["content"]
            doc_type = doc.get("type", "")

            # 文档元数据
            self.documents[doc_id] = {
                "title": title,
                "type": doc_type,
                "section": doc.get("section", ""),
                "page": doc.get("page", 0)
            }

            # 分块
            paragraphs = content.split("\n\n")
            for i, para in enumerate(paragraphs):
                if not para.strip():
                    continue

                chunk = DocumentChunk(
                    document_id=doc_id,
                    title=title,
                    content=para.strip(),
                    section=doc.get("section", ""),
                    page=doc.get("page", 0),
                    chunk_id=i,
                    doc_type=doc_type
                )
                self.chunks.append(chunk)
                all_texts.append(para.strip())

        # 编码
        logger.info(f"Encoding {len(all_texts)} chunks...")
        embeddings = self.encoder.encode(all_texts, show_progress=True)

        # 构建检索器
        self.retriever = HybridRetriever(self.encoder)
        self.retriever.build_index(self.chunks, embeddings)

        logger.info(f"Loaded {len(documents)} documents, {len(self.chunks)} chunks")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_rerank: bool = True
    ) -> List[SearchResult]:
        """检索"""
        if not self.retriever:
            raise RuntimeError("Knowledge base not initialized")

        # 编码查询
        query_embedding = self.encoder.encode_single(query)

        # 检索
        results = self.retriever.search(
            query_embedding,
            query,
            top_k=top_k * 2,  # 预留重排序空间
            filters=filters,
            alpha=0.7
        )

        # 重排序
        if use_rerank and self.reranker and len(results) > 0:
            results = self.reranker.rerank(query, results, top_k=top_k)

        return results[:top_k]

    def add_document(self, document: Dict):
        """增量添加文档"""
        # TODO: 实现增量更新
        pass


# ============== 答案生成 ==============
class AnswerGenerator:
    """答案生成器 - 支持多LLM"""

    def __init__(
        self,
        llm_provider: str = "mock",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Prompt 模板
        self.prompt_template = """你是一个企业文档问答助手。根据以下参考资料回答用户问题。

参考资料：
{context}

用户问题：{query}

要求：
1. 回答要简洁准确，基于提供的资料
2. 必须引用原文片段，格式：[文档ID]
3. 如果资料无法回答，请明确说明"未找到相关信息"

回答："""

    def generate(
        self,
        query: str,
        context_chunks: List[SearchResult],
        require_citation: bool = True
    ) -> tuple[str, List[Dict]]:
        """
        生成答案

        Returns:
            (答案, 引用列表)
        """
        # 构建上下文
        context_text = "\n\n".join([
            f"[文档{i+1} - {chunk.document_id}]\n{chunk.content[:300]}"
            for i, chunk in enumerate(context_chunks[:3])
        ])

        # 构建 Prompt
        prompt = self.prompt_template.format(
            context=context_text,
            query=query
        )

        # 调用 LLM
        answer = self._call_llm(prompt)

        # 提取引用
        citations = []
        for chunk in context_chunks:
            citations.append({
                "document_id": chunk.document_id,
                "title": chunk.title,
                "section": chunk.section,
                "page": chunk.page,
                "similarity": chunk.rerank_score or chunk.similarity
            })

        return answer, citations

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        # 模拟响应
        if "应急" in prompt and "处置" in prompt:
            return """根据《工业设备故障应急处置预案》，应急处置流程分为以下4步：

**1. 告警确认与停机**
收到故障告警后，当班人员应立即确认告警真实性，确认后立即按下紧急停机按钮，停止设备运行。

**2. 故障定位与隔离**
通过查看传感器数据分析故障原因，定位故障点。将故障设备与生产线隔离，防止故障扩散。

**3. 抢修与恢复**
根据故障类型更换损坏部件，调整设备参数。完成维修后启动设备试运行，确认运行正常后恢复生产。

**4. 复盘与记录**
详细记录故障发生原因、处理过程和预防措施。组织技术团队进行复盘，优化设备维护计划。

详情请参考文档 EP-IND-003《工业设备故障应急处置预案》第3章。"""

        if "维护" in prompt or "保养" in prompt:
            return """根据《生产线设备运维操作手册》，日常维护要点包括：

**1. 日常检查**
- 开机前检查设备外观是否有异常
- 检查各传感器显示值是否在正常范围
- 确认润滑系统油位正常

**2. 运行监测**
- 监控振动、温度、电流等关键参数
- 记录异常数据并及时上报

**3. 定期保养**
- 每周进行一次全面检查
- 每月进行预防性维护
- 每季度进行深度保养

详情请参考文档 OM-IND-012《生产线设备运维操作手册》第2章。"""

        return "感谢您的查询，请提供更具体的问题以便我为您提供准确信息。"


# ============== 文档推荐 ==============
class DocumentRecommender:
    """文档推荐器 - 基于内容相关性和协同过滤"""

    def __init__(self, knowledge_base: DocumentKnowledgeBase):
        self.kb = knowledge_base

        # 构建文档图
        self.doc_graph: Dict[str, Set[str]] = defaultdict(set)

    def build_graph(self):
        """构建文档关联图"""
        # 基于元数据构建关联
        type_groups = defaultdict(list)
        for doc_id, meta in self.kb.documents.items():
            type_groups[meta["type"]].append(doc_id)

        # 同类型文档互相关联
        for doc_ids in type_groups.values():
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    self.doc_graph[doc_ids[i]].add(doc_ids[j])
                    self.doc_graph[doc_ids[j]].add(doc_ids[i])

    def get_related_docs(
        self,
        doc_ids: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """获取关联文档"""
        if not self.doc_graph:
            self.build_graph()

        # 收集关联文档及权重
        related_scores: Dict[str, float] = defaultdict(float)

        for doc_id in doc_ids:
            for related_id in self.doc_graph.get(doc_id, []):
                if related_id not in doc_ids:
                    related_scores[related_id] += 1.0

        # 排序
        sorted_docs = sorted(related_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for related_id, score in sorted_docs[:top_k]:
            meta = self.kb.documents.get(related_id, {})
            results.append({
                "document_id": related_id,
                "title": meta.get("title", ""),
                "similarity": score / len(doc_ids)
            })

        return results


# ============== 缓存 ==============
class QueryCache:
    """查询缓存 - 多级缓存"""

    def __init__(self, ttl: int = 300, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, tuple] = {}
        self.access_order: List[str] = []
        self.lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    # 更新访问顺序
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    return value
                else:
                    del self.cache[key]
                    self.access_order.remove(key)
        return None

    def set(self, key: str, value: Any):
        """设置缓存"""
        with self.lock:
            # 淘汰
            while len(self.cache) >= self.max_size and self.access_order:
                oldest = self.access_order.pop(0)
                if oldest in self.cache:
                    del self.cache[oldest]

            self.cache[key] = (value, time.time())
            self.access_order.append(key)

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


# ============== 权限管理 ==============
class PermissionManager:
    """权限管理器"""

    def __init__(self):
        # 用户权限缓存
        self.user_permissions: Dict[str, List[str]] = {}
        self._init_default_permissions()

    def _init_default_permissions(self):
        """初始化默认权限"""
        self.user_permissions = {
            "EMP-10086": ["emergency_plan", "operation_manual", "technical_spec", "safety_regulation"],
            "EMP-10087": ["operation_manual", "technical_spec"],
            "EMP-10088": ["emergency_plan", "safety_regulation"],
            "EMP-10089": ["operation_manual"],
            "EMP-10090": ["technical_spec", "safety_regulation"]
        }

    def check_permission(
        self,
        user_id: str,
        document_types: List[str]
    ) -> Optional[List[str]]:
        """检查权限"""
        user_perms = self.user_permissions.get(user_id, [])

        if not document_types:
            return user_perms

        # 返回有权限的类型
        accessible = [dt for dt in document_types if dt in user_perms]
        return accessible if accessible else None

    def grant_permission(self, user_id: str, document_type: str):
        """授予权限"""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = []
        if document_type not in self.user_permissions[user_id]:
            self.user_permissions[user_id].append(document_type)

    def revoke_permission(self, user_id: str, document_type: str):
        """撤销权限"""
        if user_id in self.user_permissions:
            self.user_permissions[user_id] = [
                dt for dt in self.user_permissions[user_id]
                if dt != document_type
            ]


# ============== 主 Agent ==============
class EnterpriseDocumentQueryAgent:
    """
    企业资料智能查询 Agent (完善版)
    支持：混合检索、重排序、多级缓存、查询改写
    """

    def __init__(
        self,
        encoder_model: str = "paraphrase-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        enable_cache: bool = True,
        cache_ttl: int = 300
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.enable_cache = enable_cache

        # 初始化组件
        self.encoder = DocumentEncoder(encoder_model, device=self.device)
        self.reranker = Reranker(reranker_model, device=self.device)
        self.knowledge_base = DocumentKnowledgeBase(self.encoder, self.reranker)
        self.answer_generator = AnswerGenerator()
        self.recommender = DocumentRecommender(self.knowledge_base)
        self.permission_manager = PermissionManager()
        self.query_rewriter = QueryRewriter()

        # 缓存
        self.cache = QueryCache(ttl=cache_ttl) if enable_cache else None

        # 统计
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0
        }
        self.stats_lock = Lock()

        logger.info(f"Agent initialized on {self.device}")

    def load_documents(self, documents: List[Dict]):
        """加载文档"""
        self.knowledge_base.load_documents(documents)
        self.recommender.build_graph()

    def _get_cache_key(self, user_id: str, query: str, filters: Dict) -> str:
        """生成缓存键"""
        key_str = f"{user_id}:{query}:{json.dumps(filters, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    @synchronized
    def query(self, user_query: UserQuery) -> QueryResult:
        """
        查询入口

        Returns:
            QueryResult
        """
        start_time = time.time()
        self._update_stats("total_queries", 1)

        # 检查缓存
        if self.cache:
            filters = {"document_type": user_query.document_type}
            cache_key = self._get_cache_key(user_query.user_id, user_query.query, filters)
            cached_result = self.cache.get(cache_key)

            if cached_result:
                self._update_stats("cache_hits", 1)
                logger.info(f"Cache hit")
                return cached_result

        # 1. 权限校验
        permission = self.permission_manager.check_permission(
            user_query.user_id,
            user_query.document_type
        )

        if permission is None:
            return QueryResult(
                status="permission_denied",
                answer=None,
                citations=[],
                related_docs=[],
                message="您无权访问该类型文档，请联系管理员"
            )

        # 2. 查询改写
        rewrite_result = self.query_rewriter.rewrite(user_query.query)

        # 3. 构建过滤条件
        filters = rewrite_result.filters.copy()
        if user_query.document_type:
            filters["document_type"] = permission

        # 4. 文档检索
        try:
            search_results = self.knowledge_base.retrieve(
                rewrite_result.expanded,
                top_k=5,
                filters=filters,
                use_rerank=True
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return QueryResult(
                status="retrieval_error",
                answer=None,
                citations=[],
                related_docs=[],
                message=f"检索失败: {str(e)}"
            )

        if not search_results:
            return QueryResult(
                status="no_result",
                answer=None,
                citations=[],
                related_docs=[],
                message="未找到相关资料，请调整查询关键词或确认权限"
            )

        # 5. 答案生成
        try:
            answer, citations = self.answer_generator.generate(
                user_query.query,
                search_results,
                require_citation=True
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return QueryResult(
                status="generation_error",
                answer=None,
                citations=[],
                related_docs=[],
                message=f"答案生成失败: {str(e)}"
            )

        # 6. 文档推荐
        doc_ids = [r.document_id for r in search_results]
        related_docs = self.recommender.get_related_docs(doc_ids)

        # 构建结果
        result = QueryResult(
            status="success",
            answer=answer,
            citations=citations,
            related_docs=related_docs,
            metadata={
                "query_rewrite": {
                    "original": rewrite_result.original,
                    "expanded": rewrite_result.expanded,
                    "keywords": rewrite_result.keywords
                },
                "latency_ms": (time.time() - start_time) * 1000
            }
        )

        # 缓存结果
        if self.cache and result.status == "success":
            self.cache.set(cache_key, result)

        # 更新统计
        latency = (time.time() - start_time) * 1000
        self._update_latency(latency)

        return result

    def _update_stats(self, key: str, value: int):
        """更新统计"""
        with self.stats_lock:
            self.stats[key] += value

    def _update_latency(self, latency: float):
        """更新延迟统计"""
        with self.stats_lock:
            n = self.stats["total_queries"]
            old_avg = self.stats["avg_latency_ms"]
            self.stats["avg_latency_ms"] = (old_avg * (n - 1) + latency) / n

    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.stats_lock:
            return self.stats.copy()

    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")


# ============== 导出 ==============
def create_agent(
    encoder_model: str = "paraphrase-MiniLM-L6-v2",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = "cuda"
) -> EnterpriseDocumentQueryAgent:
    """创建 Agent"""
    return EnterpriseDocumentQueryAgent(
        encoder_model=encoder_model,
        reranker_model=reranker_model,
        device=device
    )


# 示例
if __name__ == "__main__":
    # 文档
    documents = [
        {
            "id": "EP-IND-003",
            "title": "工业设备故障应急处置预案",
            "type": "emergency_plan",
            "content": """工业设备故障应急处置流程分为4步：

1. 告警确认与停机
收到故障告警后，当班人员应立即确认告警真实性，确认后立即按下紧急停机按钮，停止设备运行。

2. 故障定位与隔离
通过查看传感器数据分析故障原因，定位故障点。将故障设备与生产线隔离，防止故障扩散。

3. 抢修与恢复
根据故障类型更换损坏部件，调整设备参数。完成维修后启动设备试运行，确认运行正常后恢复生产。

4. 复盘与记录
详细记录故障发生原因、处理过程和预防措施。组织技术团队进行复盘，优化设备维护计划。""",
            "section": "第3章 处置流程",
            "page": 5
        },
        {
            "id": "OM-IND-012",
            "title": "生产线设备运维操作手册",
            "type": "operation_manual",
            "content": """生产线设备日常维护要点：

1. 日常检查
- 开机前检查设备外观是否有异常
- 检查各传感器显示值是否在正常范围
- 确认润滑系统油位正常

2. 运行监测
- 监控振动、温度、电流等关键参数
- 记录异常数据并及时上报

3. 定期保养
- 每周进行一次全面检查
- 每月进行预防性维护
- 每季度进行深度保养""",
            "section": "第2章 日常维护",
            "page": 10
        }
    ]

    # 创建 Agent
    agent = create_agent()
    agent.load_documents(documents)

    # 查询
    result = agent.query(UserQuery(
        user_id="EMP-10086",
        query="工业设备故障应急处置的流程是什么？",
        document_type=["emergency_plan", "operation_manual"]
    ))

    print(f"Status: {result.status}")
    print(f"Answer: {result.answer}")
    print(f"Stats: {agent.get_stats()}")
