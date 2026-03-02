# RAG 文档处理流程设计

## 一、整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                            API 接口层                                  │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  POST /api/v1/documents/upload                                │  │
│  │  DELETE /api/v1/documents/{document_id}                       │  │
│  │  GET /api/v1/documents/{document_id}/chunks                    │  │
│  │  GET /api/v1/rag/search                                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        文档处理服务层                                  │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  DocumentProcessingService                                  │  │
│  │  - process_document(file) -> DocumentResult                 │  │
│  │  - delete_document(document_id)                            │  │
│  │  - search_documents(query) -> SearchResult                  │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ 文件解析     │          │ 文档切分     │          │ 语义精炼     │
│ (新文件)     │─────────▶│ (已存在)     │─────────▶│ + 特征提取   │
│              │          │              │          │ (新文件)     │
└──────────────┘          └──────────────┘          └──────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ 读取文本     │          │ 生成ID       │          │ JSON封装     │
│ Word/PDF/MD  │          │ document_id  │          │              │
│              │          │ chunk_id     │          │              │
└──────────────┘          └──────────────┘          └──────────────┘
                                                                 │
                                                                 ▼
                                                    ┌──────────────┐
                                                    │ 向量化       │
                                                    │ (已存在)     │
                                                    └──────────────┘
                                                            │
                                      ┌─────────────────────┼─────────────────────┐
                                      ▼                                           ▼
                              ┌──────────────┐                            ┌──────────────┐
                              │ Milvus存储   │                            │ ES存储       │
                              │ (已存在)     │                            │ (已存在)     │
                              └──────────────┘                            └──────────────┘
```

---

## 二、文件清单

### 新增文件

| 文件路径 | 说明 |
|---------|------|
| `bz_agent/api/__init__.py` | API模块初始化 |
| `bz_agent/api/document_routes.py` | 文档相关路由（上传、删除、查询） |
| `bz_agent/api/rag_routes.py` | RAG搜索路由 |
| `bz_agent/api/schemas.py` | API请求/响应的Pydantic模型 |
| `bz_agent/rag/file_parser.py` | 文件解析器（Word/PDF/Markdown） |
| `bz_agent/rag/document_processor.py` | 文档处理服务（核心编排） |
| `bz_agent/rag/semantic_refiner.py` | 语义精炼和特征提取 |
| `bz_agent/rag/models.py` | 数据模型定义 |
| `api/main.py` | FastAPI应用入口 |

### 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `requirements.txt` | 添加FastAPI、uvicorn、python-docx、markdown等依赖 |
| `config/application.yaml` | 添加文档处理相关配置（文件存储路径、模型选择等） |

### 无需修改

| 文件路径 | 说明 |
|---------|------|
| `bz_agent/rag/document_splitter.py` | 已实现，直接使用 |
| `bz_agent/rag/embedding_data_handler.py` | 已实现，直接使用 |
| `bz_agent/rag/save_embedding_to_milvus.py` | 已实现，直接使用 |
| `bz_agent/rag/bm25_es_search.py` | 已实现，可直接扩展写入功能 |

---

## 三、核心流程设计

### 3.1 文档处理完整流程

```
1. 接收文件上传请求
   │
   ├─ 验证文件类型（Word/PDF/Markdown）
   ├─ 生成唯一 document_id（使用现有 snowflake）
   └─ 保存原始文件到指定目录
   │
2. 解析文件内容
   │
   ├─ Word: 使用 python-docx 提取文本
   ├─ PDF: 使用 pypdf/pymupdf 提取文本
   └─ Markdown: 直接读取 + 可选提取标题结构
   │
3. 文档切分
   │
   ├─ 使用 DocumentSplitter
   ├─ 每个chunk分配唯一 chunk_id
   ├─ 记录元数据（document_id, chunk_index, original_content）
   └─ 返回 List[ChunkInfo]
   │
4. 语义精炼和特征提取（并行处理每个chunk）
   │
   ├─ 调用 LLM 精炼摘要
   ├─ 提取关键词/特征词
   ├─ 提取实体（可选）
   └─ 返回 RefinementResult
   │
5. JSON封装
   │
   └─ 组装最终数据结构
      {
        "document_id": "...",
        "chunk_id": "...",
        "chunk_index": 0,
        "original_content": "...",
        "refined_summary": "...",
        "keywords": ["...", "..."],
        "entities": ["...", "..."],
        "metadata": {...}
      }
   │
6. 向量化（使用现有 embedding 模型）
   │
   └─ 生成 embedding 向量
   │
7. 双写存储
   │
   ├─ Milvus: {id, document_id, chunk_content, embedding, metadata}
   └─ Elasticsearch: {document_id, chunk_id, content, refined_summary, keywords, entities}
   │
8. 返回处理结果
```

### 3.2 数据结构设计

```python
# 文档上传请求
DocumentUploadRequest:
    file: UploadFile
    split_strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    enable_refinement: bool = True

# 文档处理结果
DocumentProcessResult:
    document_id: str
    filename: str
    chunk_count: int
    status: "success" | "partial" | "failed"
    chunks: List[ChunkResult]
    errors: List[str]

# 切分块信息
ChunkInfo:
    document_id: str
    chunk_id: str
    chunk_index: int
    original_content: str
    metadata: Dict

# 语义精炼结果
RefinementResult:
    chunk_id: str
    refined_summary: str
    keywords: List[str]
    entities: List[str]

# 最终存储数据
ChunkDocument:
    document_id: str
    chunk_id: str
    chunk_index: int
    original_content: str
    refined_summary: str
    keywords: List[str]
    entities: List[str]
    embedding: List[float]
    metadata: Dict
```

---

## 四、接口设计

### 4.1 文档上传

```
POST /api/v1/documents/upload

Request:
  Content-Type: multipart/form-data
  Body:
    file: binary
    split_strategy?: "recursive" | "markdown_header" | "semantic" | "hybrid"
    chunk_size?: int (default: 500)
    chunk_overlap?: int (default: 50)
    enable_refinement?: boolean (default: true)

Response 200:
  {
    "code": 0,
    "message": "success",
    "data": {
      "document_id": "1234567890123456789",
      "filename": "document.pdf",
      "chunk_count": 15,
      "status": "success",
      "chunks": [...]
    }
  }
```

### 4.2 删除文档

```
DELETE /api/v1/documents/{document_id}

Response 200:
  {
    "code": 0,
    "message": "success",
    "data": {
      "document_id": "1234567890123456789",
      "deleted_chunks": 15
    }
  }
```

### 4.3 查询文档详情

```
GET /api/v1/documents/{document_id}

Response 200:
  {
    "code": 0,
    "message": "success",
    "data": {
      "document_id": "1234567890123456789",
      "filename": "document.pdf",
      "upload_time": "2026-03-02T10:00:00",
      "chunk_count": 15,
      "chunks": [...]
    }
  }
```

### 4.4 RAG搜索

```
GET /api/v1/rag/search

Query params:
  query: string
  top_k?: int (default: 10)
  use_rerank?: boolean (default: false)
  filter?: string (ES filter DSL)

Response 200:
  {
    "code": 0,
    "message": "success",
    "data": {
      "query": "用户问题",
      "results": [
        {
          "document_id": "1234567890123456789",
          "chunk_id": "1234567890123456789_0",
          "content": "...",
          "refined_summary": "...",
          "score": 0.95,
          "metadata": {...}
        }
      ]
    }
  }
```

---

## 五、配置设计

### 5.1 application.yaml 新增配置

```yaml
# 文档存储配置
document:
  storage_path: "./data/documents"  # 原始文件存储路径
  max_file_size: 10485760  # 10MB
  allowed_extensions: [".pdf", ".docx", ".md", ".txt"]

# 语义精炼配置
semantic_refinement:
  enabled: true
  llm_model: "basic"  # 使用哪个LLM模型
  max_keywords: 10
  max_entities: 5
  summary_max_length: 200

# API配置
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
```

### 5.2 requirements.txt 新增依赖

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.20
python-docx==1.1.3
pymupdf==1.24.0  # 或 pypdf
markdown==3.7
```

---

## 六、依赖关系图

```
api/main.py
  │
  ├─► bz_agent/api/document_routes.py
  │       │
  │       ├─► bz_agent/rag/document_processor.py
  │       │       │
  │       │       ├─► bz_agent/rag/file_parser.py (NEW)
  │       │       ├─► bz_agent/rag/document_splitter.py (EXISTING)
  │       │       ├─► bz_agent/rag/semantic_refiner.py (NEW)
  │       │       ├─► bz_agent/rag/embedding_data_handler.py (EXISTING)
  │       │       ├─► bz_agent/rag/save_embedding_to_milvus.py (EXISTING)
  │       │       └─► bz_agent/rag/bm25_es_search.py (EXISTING - extend)
  │       │
  │       └─► bz_agent/api/schemas.py (NEW)
  │
  └─► bz_agent/api/rag_routes.py
          │
          └─► bz_agent/rag/bm25_es_search.py (EXISTING - extend)
```

---

## 七、实现优先级

| 优先级 | 任务 | 说明 |
|-------|------|------|
| P0 | 文件解析器 | 支持 Word/PDF/Markdown 读取 |
| P0 | 文档处理器 | 核心流程编排 |
| P0 | API路由 | 上传、删除、查询接口 |
| P0 | Milvus存储集成 | 存储向量数据 |
| P1 | 语义精炼 | LLM摘要和关键词提取 |
| P1 | Elasticsearch存储 | 存储结构化数据 |
| P2 | Rerank集成 | 搜索结果重排序 |
| P2 | 批量上传 | 支持多文件上传 |
| P3 | 文档更新 | 支持增量更新 |

---

## 八、潜在问题和考虑

1. **大文件处理**：超过100MB的文件可能需要流式处理或异步队列
2. **并发处理**：同时上传多个文件时需要考虑资源限制
3. **错误恢复**：部分chunk处理失败时的处理策略
4. **版本控制**：同一document_id上传多次时的版本管理
5. **索引更新**：Milvus和ES双写的原子性保证
6. **模型选择**：语义精炼使用哪个LLM（basic还是reasoning）
7. **成本控制**：调用LLM的token消耗和成本估算

---

*设计文档创建时间: 2026-03-02*
