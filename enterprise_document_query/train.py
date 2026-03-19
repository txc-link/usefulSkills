"""
企业资料智能查询 - 完整训练脚本
支持：对比学习微调、知识库构建、评估指标
"""
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import (
    SentenceTransformer, InputExample, evaluation, losses
)
from sentence_transformers import CrossEncoder
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import defaultdict
import time
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== 数据集 ==============
class RetrievalDataset(Dataset):
    """检索数据集"""

    def __init__(self, data_path: str, split: str = "train"):
        self.data = self._load_data(data_path)
        self.samples = self._create_samples(split)

    def _load_data(self, path: str) -> list:
        path = Path(path)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif path.suffix == ".csv":
            return pd.read_csv(path).to_dict("records")
        raise ValueError(f"Unsupported: {path.suffix}")

    def _create_samples(self, split: str) -> list:
        pairs = []

        for item in self.data:
            query = item.get("query", "")
            positives = item.get("positive_docs", [])
            negatives = item.get("negative_docs", [])

            for pos in positives:
                pairs.append({
                    "query": query,
                    "document": pos,
                    "label": 1.0
                })

            for neg in negatives[:3]:  # 每个正样本配3个负样本
                pairs.append({
                    "query": query,
                    "document": neg,
                    "label": 0.0
                })

        # 划分
        n = len(pairs)
        if split == "train":
            return pairs[:int(n * 0.7)]
        elif split == "val":
            return pairs[int(n * 0.7):int(n * 0.85)]
        return pairs[int(n * 0.85):]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return InputExample(texts=[s["query"], s["document"]], label=s["label"])


class CrossEncoderDataset(Dataset):
    """Cross-Encoder 训练数据集"""

    def __init__(self, data_path: str, split: str = "train"):
        self.data = self._load_data(data_path)
        self.samples = self._create_samples(split)

    def _load_data(self, path: str) -> list:
        path = Path(path)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _create_samples(self, split: str) -> list:
        pairs = []
        for item in self.data:
            query = item.get("query", "")
            for pos in item.get("positive_docs", []):
                pairs.append((query, pos, 1))
            for neg in item.get("negative_docs", [])[:2]:
                pairs.append((query, neg, 0))

        n = len(pairs)
        if split == "train":
            return pairs[:int(n * 0.7)]
        elif split == "val":
            return pairs[int(n * 0.7):int(n * 0.85)]
        return pairs[int(n * 0.85):]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, d, label = self.samples[idx]
        return q, d, torch.tensor(label, dtype=torch.float)


# ============== 数据生成 ==============
def generate_retrieval_data(
    num_queries: int = 500,
    docs_per_query: int = 5,
    output_path: str = "data/retrieval_data.json"
):
    """生成检索数据"""
    np.random.seed(42)

    docs = {
        "emergency_plan": [
            "应急处置流程：1. 告警确认 2. 故障定位 3. 抢修恢复 4. 复盘记录",
            "应急预案要求：发现异常立即停机，通知值班人员",
            "应急响应级别分为红色、橙色、黄色、蓝色四级",
            "应急演练每季度进行一次，包含桌面推演和实战演练",
            "应急物资储备应满足72小时连续作业需求",
            "应急指挥中心负责统一调度和协调处置工作"
        ],
        "operation_manual": [
            "设备操作规程：开机前检查各系统状态，确认无异常后方可启动",
            "日常维护：每班次进行设备点检，记录运行参数",
            "定期保养：每周进行润滑、紧固、清洁工作",
            "操作人员必须持证上岗，严格遵守操作规程",
            "设备运行参数：温度<60°C，振动<5mm/s，压力0.4-0.6MPa",
            "停机操作顺序：先停负载，再停主机，最后断电"
        ],
        "technical_spec": [
            "技术规格：额定功率500kW，转速1500rpm，电压380V",
            "设备精度等级：ISO 5级，表面粗糙度Ra1.6",
            "主要材质：HT250铸铁，轴套采用青铜衬套",
            "设计寿命：100000小时，年故障率<2%",
            "备件清单：轴承、密封圈、润滑油、过滤器等",
            "安装要求：基础平整度<0.05mm/m，对中精度<0.02mm"
        ]
    }

    queries = [
        "设备故障应急处置流程是什么？",
        "如何进行日常设备维护？",
        "技术规格参数有哪些？",
        "应急预案响应级别如何划分？",
        "操作人员需要什么资质？",
        "设备运行参数标准是多少？",
        "备件更换周期是多久？",
        "故障排查步骤有哪些？",
        "应急演练多久进行一次？",
        "设备精度等级是多少？"
    ]

    data = []
    for i in range(num_queries):
        query = np.random.choice(queries)
        doc_type = np.random.choice(list(docs.keys()))

        positives = np.random.choice(docs[doc_type], size=min(docs_per_query, len(docs[doc_type])), replace=False).tolist()

        negatives = []
        for ot in docs.keys():
            if ot != doc_type:
                negatives.extend(docs[ot])

        data.append({
            "query": query,
            "positive_docs": positives,
            "negative_docs": negatives
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Generated {num_queries} queries -> {output_path}")
    return data


# ============== 训练器 ==============
class BiEncoderTrainer:
    """双编码器训练器"""

    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        device: str = "cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def prepare(self, train_samples: List, val_samples: List = None):
        """准备训练数据"""
        self.train_examples = [InputExample(s["query"], s["document"], s["label"]) for s in train_samples]
        self.train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=32)

        self.train_loss = losses.ContrastiveLoss(self.model)

        if val_samples:
            self.val_examples = [InputExample(s["query"], s["document"], s["label"]) for s in val_samples]
        else:
            self.val_examples = None

    def train(
        self,
        output_path: str = "output/bi_encoder",
        epochs: int = 10,
        warmup_steps: int = 100
    ):
        """训练"""
        evaluators = []
        if self.val_examples:
            evaluators.append(evaluation.BinaryClassificationEvaluator(
                texts1=[e.texts[0] for e in self.val_examples],
                texts2=[e.texts[1] for e in self.val_examples],
                labels=[e.label for e in self.val_examples]
            ))

        self.model.fit(
            train_objectives=[(self.train_dataloader, self.train_loss)],
            evaluator=evaluation.SequentialEvaluator(evaluators) if evaluators else None,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            evaluation_steps=500 if self.val_examples else 0,
            show_progress_bar=True
        )

        print(f"Model saved to {output_path}")


class CrossEncoderTrainer:
    """Cross-Encoder 训练器"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, max_length=512, device=self.device)

    def prepare(self, train_data: List, val_data: List = None):
        """准备数据"""
        self.train_data = train_data

        if val_data:
            self.val_data = val_data
        else:
            self.val_data = None

    def train(
        self,
        output_path: str = "output/cross_encoder",
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """训练"""
        train_samples = []

        for q, d, label in self.train_data:
            train_samples.append({
                "texts": [q, d],
                "label": label
            })

        self.model.fit(
            train_samples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        self.model.save(output_path)
        print(f"Cross-Encoder saved to {output_path}")


# ============== 知识库构建 ==============
class KnowledgeBaseBuilder:
    """知识库构建器"""

    def __init__(self, encoder_model: str = "paraphrase-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(encoder_model)
        self.documents = []
        self.chunks = []
        self.metadata: Dict[str, Dict] = {}

    def load_documents(
        self,
        documents: List[Dict],
        chunk_size: int = 500,
        overlap: int = 50
    ):
        """加载并分块文档"""
        self.chunks = []

        for doc in documents:
            doc_id = doc.get("id", "")
            title = doc.get("title", "")
            content = doc.get("content", "")
            doc_type = doc.get("type", "")
            section = doc.get("section", "")
            page = doc.get("page", 0)

            self.metadata[doc_id] = {
                "title": title,
                "type": doc_type,
                "section": section,
                "page": page
            }

            # 分块
            for i, para in enumerate(content.split("\n\n")):
                if not para.strip():
                    continue

                self.chunks.append({
                    "document_id": doc_id,
                    "title": title,
                    "content": para.strip(),
                    "section": section,
                    "page": page,
                    "doc_type": doc_type,
                    "chunk_id": i
                })

        print(f"Loaded {len(documents)} docs -> {len(self.chunks)} chunks")

    def build_index(
        self,
        index_type: str = "IVF",
        nlist: int = 100
    ):
        """构建向量索引"""
        texts = [c["content"] for c in self.chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        dim = embeddings.shape[1]
        embeddings = embeddings.astype('float32')

        if index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(embeddings)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        print(f"Built {index_type} index: {self.index.ntotal} vectors")

    def save(self, path: str):
        """保存知识库"""
        import pickle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存索引
        faiss.write_index(self.index, str(path / "index.faiss"))

        # 保存文档
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        # 保存元数据
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        # 保存模型配置
        config = {
            "encoder_model": self.encoder.model_name,
            "num_chunks": len(self.chunks),
            "embedding_dim": self.encoder.get_sentence_embedding_dimension()
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Knowledge base saved to {path}")

    def load(self, path: str):
        """加载知识库"""
        import pickle

        path = Path(path)

        # 索引
        self.index = faiss.read_index(str(path / "index.faiss"))

        # 文档
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        # 元数据
        with open(path / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        print(f"Loaded {len(self.chunks)} chunks from {path}")


# ============== 评估 ==============
class RetrievalEvaluator:
    """检索评估器"""

    def __init__(self, knowledge_base: KnowledgeBaseBuilder):
        self.kb = knowledge_base
        self.encoder = SentenceTransformer(self.kb.encoder.model_name)

    def evaluate(
        self,
        test_queries: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """评估检索性能"""

        metrics = {f"precision@{k}": [] for k in k_values}
        metrics[f"mrr"] = []  # Mean Reciprocal Rank
        metrics[f"ndcg@{k}"] = {k: [] for k in k_values}

        for item in test_queries:
            query = item["query"]
            positives = set(item.get("positive_docs", []))
            pos_set = set(positives) if positives else set()

            if not pos_set:
                continue

            # 检索
            q_emb = self.encoder.encode_single(query)
            q_emb = q_emb.reshape(1, -1).astype('float32')

            scores, indices = self.kb.index.search(q_emb, max(k_values))

            # 计算指标
            retrieved_docs = [self.kb.chunks[i]["content"] for i in indices[0]]

            for k in k_values:
                retrieved_k = retrieved_docs[:k]
                hits = sum(1 for d in retrieved_k if d in pos_set)
                precision = hits / k
                metrics[f"precision@{k}"].append(precision)

            # MRR
            rank = None
            for i, d in enumerate(retrieved_docs):
                if d in pos_set:
                    rank = i + 1
                    break
            if rank:
                metrics["mrr"].append(1.0 / rank)
            else:
                metrics["mrr"].append(0)

            # NDCG
            for k in k_values:
                dcg = 0
                for i, d in enumerate(retrieved_docs[:k]):
                    if d in pos_set:
                        dcg += 1.0 / (i + 1)

                idcg = sum(1.0 / (i + 1) for i in range(min(k, len(pos_set))))
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics[f"ndcg@{k}"][k].append(ndcg)

        # 平均
        results = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                results[k] = {kk: np.mean(vv) for kk, vv in v.items()}
            else:
                results[k] = np.mean(v)

        return results


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description="Train Document Retrieval")
    parser.add_argument("--data_path", type=str, default="data/retrieval_data.json")
    parser.add_argument("--generate_data", action="store_true")
    parser.add_argument("--num_queries", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="paraphrase-MiniLM-L6-v2")
    parser.add_argument("--output_path", type=str, default="output/model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 生成数据
    if args.generate_data:
        generate_retrieval_data(args.num_queries, output_path=args.data_path)

    # 加载数据
    dataset = RetrievalDataset(args.data_path)
    train_samples = dataset.samples
    print(f"Loaded {len(train_samples)} training samples")

    # 训练双编码器
    trainer = BiEncoderTrainer(args.model_name, args.device)
    trainer.prepare(train_samples)
    trainer.train(args.output_path, args.epochs)

    print("Bi-Encoder training complete!")


def build_kb():
    """构建知识库"""
    parser = argparse.ArgumentParser(description="Build Knowledge Base")
    parser.add_argument("--documents", type=str, default="data/documents.json")
    parser.add_argument("--model_name", type=str, default="paraphrase-MiniLM-L6-v2")
    parser.add_argument("--output_path", type=str, default="output/knowledge_base")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--index_type", type=str, default="IVF")
    args = parser.parse_args()

    # 加载文档
    with open(args.documents, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # 构建
    builder = KnowledgeBaseBuilder(args.model_name)
    builder.load_documents(documents, args.chunk_size)
    builder.build_index(args.index_type)
    builder.save(args.output_path)


def evaluate_retrieval():
    """评估检索"""
    parser = argparse.ArgumentParser(description="Evaluate Retrieval")
    parser.add_argument("--kb_path", type=str, default="output/knowledge_base")
    parser.add_argument("--test_data", type=str, default="data/retrieval_data.json")
    args = parser.parse_args()

    # 加载知识库
    kb = KnowledgeBaseBuilder()
    kb.load(args.kb_path)

    # 加载测试数据
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 评估
    evaluator = RetrievalEvaluator(kb)
    results = evaluator.evaluate(test_data)

    print("\n=== Retrieval Evaluation Results ===")
    for metric, value in results.items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "build_kb":
            build_kb()
        elif sys.argv[1] == "evaluate":
            evaluate_retrieval()
        else:
            main()
    else:
        main()
