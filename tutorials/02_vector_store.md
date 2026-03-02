# 第二课：向量数据库与文档处理

## 课程目标

1. 理解向量数据库的工作原理
2. 掌握 Chroma 向量数据库的使用
3. 学会文档加载与文本分块
4. 理解 MD5 去重机制

---

## 1. 向量数据库概述

### 1.1 什么是向量数据库

向量数据库是一种专门用于存储和检索高维向量数据的数据库。在 AI 领域，它主要用于：

- **语义搜索**: 通过向量相似度找到相关内容
- **知识库**: 存储文档的语义表示
- **推荐系统**: 基于向量相似度推荐相似物品

### 1.2 为什么需要向量数据库

传统数据库查询：
```sql
SELECT * FROM articles WHERE content LIKE '%扫地机器人%'
```

问题：
1. **精确匹配**: 必须包含关键词
2. **同义词问题**: "清扫" 和 "清洁" 无法匹配
3. **语义理解**: 无法理解查询的真正意图

向量数据库查询：
```python
# 查询 "智能清洁设备"
results = vector_store.similarity_search("智能清洁设备")
```

优势：
1. **语义匹配**: 不需要精确关键词
2. **相似语义**: 可以找到同义词相关文档
3. **语义理解**: 理解查询的真正含义

### 1.3 Embedding 向量

文本通过 Embedding 模型转换为向量：

```
"扫地机器人"  →  [0.12, -0.34, 0.56, 0.78, ...]
"智能吸尘器"  →  [0.11, -0.33, 0.55, 0.79, ...]  # 相似！
"天气预报"    →  [-0.88, 0.21, 0.03, -0.45, ...]  # 不相似
```

相似度计算常用：
- **余弦相似度**: cos(θ)
- **欧氏距离**: ||a - b||

---

## 2. Chroma 向量数据库

### 2.1 Chroma 简介

Chroma 是一个轻量级的向量数据库，专为 AI 应用设计：

- 纯 Python 实现，易于集成
- 支持持久化存储
- 与 LangChain 无缝集成

### 2.2 项目配置

`config/chroma.yaml`:

```yaml
collection_name: "robot_knowledge"
persist_directory: "data/chroma_db"
chunk_size: 500
chunk_overlap: 50
separators: ["\n\n", "\n", "。", "！", "？", "，", " "]
k: 5
allow_knowledge_file_type: ["txt", "pdf"]
md5_hex_store: "data/chroma_md5.txt"
```

---

## 3. 源码分析

### 3.1 VectorStoreService 类

```python
class VectorStoreService:
    def __init__(self):
        # 1. 初始化 Chroma 向量数据库
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_config["persist_directory"],
        )
        
        # 2. 初始化文本分块器
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len
        )
```

### 3.2 核心概念解读

#### Chroma 初始化

```python
self.vector_store = Chroma(
    collection_name="robot_knowledge",    # 集合名称
    embedding_function=embedding_model,  # 嵌入模型
    persist_directory="data/chroma_db",  # 持久化路径
)
```

- **collection_name**: 类似于数据库表名，用于区分不同类型的文档
- **embedding_function**: 将文本转换为向量的模型
- **persist_directory**: 数据持久化路径，关闭后数据不会丢失

#### 文本分块器

```python
self.spliter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每个 chunk 最大字符数
    chunk_overlap=50,      # 相邻 chunk 重叠字符数
    separators=["\n\n", ...],  # 分隔符优先级
    length_function=len    # 计算长度的函数
)
```

- **chunk_size**: 太大可能包含无关信息，太小可能丢失上下文
- **chunk_overlap**: 保持相邻块之间的上下文连贯性
- **separators**: 按优先级尝试分割，找到最佳分割点

### 3.3 获取检索器

```python
def get_retriever(self):
    return self.vector_store.as_retriever(search_kwargs={"k": 5})
```

- **k=5**: 每次检索返回最相似的 5 个文档
- 返回的 retriever 可以直接用于 LangChain Chain

---

## 4. 文档加载与处理

### 4.1 文档加载流程

```
TXT/PDF 文件 → 文档加载器 → Document 对象列表 → 文本分块 → 向量存储
```

### 4.2 加载器实现

```python
def txt_loader(path: str) -> list[Document]:
    """加载 TXT 文件"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 创建 Document 对象
    return [Document(
        page_content=content,
        metadata={"source": path, "type": "txt"}
    )]


def pdf_loader(path: str) -> list[Document]:
    """加载 PDF 文件（使用 PyPDFLoader）"""
    from langchain_community.document_loaders import PyPDFLoader
    
    loader = PyPDFLoader(path)
    documents = loader.load()
    
    # 添加元数据
    for doc in documents:
        doc.metadata["source"] = path
        doc.metadata["type"] = "pdf"
    
    return documents
```

### 4.3 Document 对象结构

```python
from langchain_core.documents import Document

doc = Document(
    page_content="这是文档的实际内容...",
    metadata={
        "source": "data/扫地机器人100问.pdf",
        "type": "pdf",
        "page": 1
    }
)
```

- **page_content**: 文档的文本内容
- **metadata**: 元数据（来源、类型、页码等），检索时可以用于过滤

---

## 5. MD5 去重机制

### 5.1 为什么需要去重

每次加载文档都重新处理并存储向量会导致：
1. **数据冗余**: 同一文档存储多次
2. **查询结果重复**: 可能返回重复内容
3. **存储空间浪费**

### 5.2 MD5 原理

MD5（Message-Digest Algorithm 5）是一种散列函数：

```
任意长度输入 → 128位固定长度输出（32位十六进制）
```

特点：
- 相同输入 → 相同输出
- 不同输入 → 几乎不可能相同
- 不可逆

### 5.3 去重实现

```python
def get_file_md5_hex(filepath: str) -> str:
    """计算文件的 MD5 值"""
    import hashlib
    
    md5_hash = hashlib.md5()
    
    with open(filepath, "rb") as f:
        # 分块读取，避免大文件内存问题
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest()


def check_md5_hex(md5_for_check: str) -> bool:
    """检查 MD5 是否已存在"""
    md5_file = get_abs_path(chroma_config["md5_hex_store"])
    
    if not os.path.exists(md5_file):
        return False
    
    with open(md5_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.strip() == md5_for_check:
                return True
    
    return False


def save_md5_hex(md5_for_check: str):
    """保存 MD5 到文件"""
    md5_file = get_abs_path(chroma_config["md5_hex_store"])
    
    with open(md5_file, "a", encoding="utf-8") as f:
        f.write(md5_for_check + "\n")
```

### 5.4 去重流程

```
1. 遍历数据目录中的文件
2. 计算每个文件的 MD5
3. 检查 MD5 是否已记录
   ├── 已存在 → 跳过（已加载过）
   └── 不存在 → 加载文档 → 分块 → 存储向量 → 记录 MD5
```

---

## 6. 完整加载流程

```python
def load_document(self):
    """加载文档到向量数据库"""
    
    # 1. 获取允许的文件类型
    allowed_files = listdir_with_allowed_type(
        get_abs_path(chroma_config["data_path"]),
        tuple(chroma_config["allow_knowledge_file_type"])
    )
    
    # 2. 遍历处理每个文件
    for path in allowed_files:
        # 3. 计算 MD5
        md5_hex = get_file_md5_hex(path)
        
        # 4. 检查是否已加载
        if check_md5_hex(md5_hex):
            logger.info(f"{path} 已存在，跳过")
            continue
        
        try:
            # 5. 加载文档
            documents = get_file_document(path)
            
            # 6. 文本分块
            split_documents = self.spliter.split_documents(documents)
            
            # 7. 存储到向量数据库
            self.vector_store.add_documents(split_documents)
            
            # 8. 记录 MD5
            save_md5_hex(md5_hex)
            
            logger.info(f"{path} 加载成功")
            
        except Exception as e:
            logger.error(f"{path} 加载失败: {str(e)}", exc_info=True)
```

---

## 7. 总结

本节课学习了：

1. **向量数据库**: 存储和检索语义向量的数据库
2. **Chroma**: 轻量级向量数据库，与 LangChain 无缝集成
3. **文本分块**: 将长文档分割成小块，便于检索
4. **MD5 去重**: 通过文件哈希避免重复加载
5. **Document**: LangChain 的文档数据结构

---

## 代码练习

### 练习 2.1：添加 Word 文档支持

在 `utils/file_handler.py` 中添加 `docx_loader` 函数：

```python
def docx_loader(path: str) -> list[Document]:
    """加载 Word 文档"""
    # 提示：使用 langchain_community.document_loaders 的 Docx2txtLoader
    pass
```

### 练习 2.2：自定义分块策略

创建一个新的文本分块器，按段落分割：

```python
def create_paragraph_splitter():
    """创建按段落分割的分块器"""
    # 按双换行符分割，保留段落结构
    pass
```

### 练习 2.3：元数据过滤

修改 `get_retriever` 方法，支持按文件类型过滤：

```python
def get_retriever_by_type(self, file_type: str):
    """按文件类型获取检索器"""
    # 提示：使用 filter 参数
    pass
```

---

## 下节课预告

下一课我们将学习 **RAG 服务与链式调用**，包括：
- LangChain Chain 链式调用
- PromptTemplate 模板使用
- Retriever 与 LLM 的配合

---

## 相关资源

- [Chroma 官方文档](https://docs.trychroma.com/)
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/text_splitters/recursive_text_splitter)
- [MD5 算法原理](https://zh.wikipedia.org/wiki/MD5)
