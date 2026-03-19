# 企业资料智能查询 Agent Skill

## 基本信息

```
skill_name: enterprise_document_intelligent_query
version: 1.0.0
author: AI Agent Team
description: 基于RAG技术，对接企业内部文档库，实现规章制度、技术文档、应急预案的精准检索与问答
```

## 触发短语

- 企业资料查询
- 文档检索
- 规章制度问答
- 应急预案查询

## 权限

- `read:enterprise_document_library` - 读取企业文档库
- `read:user_access_control` - 读取用户权限

## 依赖

```python
torch>=2.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
```

## 核心职责与场景边界

- **单一职责**：仅处理企业内部资料的检索与问答，不涉及外部数据或其他业务逻辑
- **场景边界**：
  - 输入：用户自然语言查询
  - 输出：文档摘要、原文引用、关联文档推荐
- **禁止行为**：
  - 不泄露未授权文档
  - 不生成虚构内容
  - 不处理非资料类请求

## 流程拆解

### Step 1: 意图识别与权限校验
- 工具调用：`user_access_api.check_permission(user_id, document_type)`
- 输出：用户权限范围、可访问文档库列表

### Step 2: 查询改写
- 扩展同义词、提取关键词
- 识别过滤条件（文档类型等）

### Step 3: 文档检索 (RAG)
- 向量检索 + BM25 混合检索
- Cross-Encoder 重排序
- 工具调用：`llamaindex.retrieve(query, top_k=5, filter=permission_filter)`
- 输出：匹配文档片段、相似度得分

### Step 4: 答案生成与引用标注
- 工具调用：`llm.generate_answer(query, context_chunks)`
- 输出：结构化答案、原文引用位置

### Step 5: 结果反馈与推荐
- 工具调用：`document_recommender.get_related_docs(doc_id_list)`
- 输出：关联文档列表、阅读建议

## 输入输出规范

### 输入示例
```json
{
  "user_id": "EMP-10086",
  "query": "工业设备故障应急处置的流程是什么？",
  "document_type": ["emergency_plan", "operation_manual"]
}
```

### 输出示例
```json
{
  "status": "success",
  "answer": "工业设备故障应急处置流程分为4步：1. 告警确认与停机；2. 故障定位与隔离；3. 抢修与恢复；4. 复盘与记录。",
  "citations": [
    {
      "document_id": "EP-IND-003",
      "title": "工业设备故障应急处置预案",
      "section": "第3章 处置流程",
      "page": 5,
      "similarity": 0.92
    }
  ],
  "related_docs": [
    {
      "document_id": "OM-IND-012",
      "title": "生产线设备运维操作手册",
      "similarity": 0.85
    }
  ]
}
```

## 支持的文档类型

| 类型 | 说明 |
|------|------|
| emergency_plan | 应急预案 |
| operation_manual | 操作手册 |
| technical_spec | 技术规格 |
| safety_regulation | 安全规程 |

## 优化与异常处理

### 性能优化
- **渐进式加载**：仅在用户发起查询时加载文档向量库
- **混合检索**：向量 + BM25 融合
- **多级缓存**：查询结果缓存 + LRU淘汰

### 异常处理

| 状态 | 场景 | 返回 |
|------|------|------|
| `no_result` | 无匹配文档 | `{"status": "no_result", "message": "未找到相关资料，请调整查询关键词或确认权限"}` |
| `permission_denied` | 权限不足 | `{"status": "permission_denied", "message": "您无权访问该类型文档，请联系管理员"}` |
| `retrieval_error` | 检索失败 | `{"status": "retrieval_error", "message": "检索服务暂时不可用"}` |
| `generation_error` | 生成失败 | `{"status": "generation_error", "message": "答案生成失败，请稍后重试"}` |

### 幻觉抑制
- 强制要求答案必须引用原文片段
- 无引用时标注为"推测内容"

## 使用方法

```python
from skill import create_agent, UserQuery

# 创建 Agent
agent = create_agent(
    encoder_model="paraphrase-MiniLM-L6-v2",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# 加载文档
documents = [
    {
        "id": "EP-IND-003",
        "title": "工业设备故障应急处置预案",
        "type": "emergency_plan",
        "content": "..."
    }
]
agent.load_documents(documents)

# 查询
result = agent.query(UserQuery(
    user_id="EMP-10086",
    query="工业设备故障应急处置的流程是什么？",
    document_type=["emergency_plan", "operation_manual"]
))

print(result.status)
print(result.answer)
print(result.citations)
```

## 训练

```bash
# 生成检索数据
python train.py --generate_data --num_queries 500

# 训练 Bi-Encoder
python train.py --data_path data/retrieval_data.json --epochs 10

# 构建知识库
python train.py build_kb --documents data/documents.json

# 评估
python train.py evaluate
```

## 架构图

```
┌─────────────┐
│  用户查询   │
└──────┬──────┘
       │
       v
┌─────────────┐     ┌─────────────────┐
│ 权限校验    │────>│ PermissionManager│
└──────┬──────┘     └─────────────────┘
       │
       v
┌─────────────┐     ┌─────────────────┐
│ 查询改写    │────>│  QueryRewriter  │
└──────┬──────┘     └─────────────────┘
       │
       v
┌─────────────┐     ┌─────────────────┐
│ 混合检索    │────>│ HybridRetriever │
│ (向量+BM25) │     └─────────────────┘
└──────┬──────┘            │
       │                   v
       │            ┌─────────────────┐
       └───────────>│    Reranker     │
                    └─────────────────┘
                           │
                           v
                    ┌─────────────────┐
                    │ AnswerGenerator │
                    └─────────────────┘
                           │
                           v
                    ┌─────────────────┐
                    │ 结果返回+推荐  │
                    └─────────────────┘
```
