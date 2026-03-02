# 第一课：模型工厂模式与 DashScope API 集成

## 课程目标

1. 理解工厂模式在 AI 项目中的应用
2. 掌握 LangChain ChatModel 接口
3. 学会集成 DashScope（阿里云）API

---

## 1. 工厂模式概述

### 1.1 什么是工厂模式

工厂模式（Factory Pattern）是一种创建型设计模式，其核心思想是**将对象的创建与使用分离**。在 AI 应用中，我们可能需要：

- 切换不同的模型（GPT、Claude、通义千问等）
- 在开发/生产环境使用不同的模型
- 统一管理模型的初始化和配置

### 1.2 为什么使用工厂模式

假设我们不用工厂模式：

```python
# ❌ 不推荐：直接在不同地方创建模型
from langchain_community.chat_models.tongyi import ChatTongyi

def rag_service():
    model = ChatTongyi(model="qwen3-max")
    # 使用 model...

def agent_service():
    model = ChatTongyi(model="qwen3-max")
    # 使用 model...
```

问题：
1. **代码重复**：每次使用都需要配置模型参数
2. **难以维护**：模型配置散落在各处
3. **切换困难**：更换模型需要修改所有使用位置

使用工厂模式后：

```python
# ✅ 推荐：统一通过工厂获取模型
from model.factory import chat_model

def rag_service():
    model = chat_model  # 统一获取
    # 使用 model...

def agent_service():
    model = chat_model  # 统一获取
    # 使用 model...
```

---

## 2. 源码分析

### 2.1 项目结构

```
model/
└── factory.py    # 模型工厂
```

### 2.2 完整源码

```python
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from utils.config_handler import rag_config


class BaseModelFactory(ABC):
    """模型工厂抽象基类"""
    
    @abstractmethod
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        """生成模型实例的抽象方法"""
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂"""
    
    def generator(self) -> Optional[BaseChatModel]:
        return ChatTongyi(model=rag_config["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    """嵌入模型工厂"""
    
    def generator(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()
```

### 2.3 核心概念解读

#### 抽象基类（ABC）

```python
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        pass
```

- **ABC**: Python 的抽象基类模块
- **@abstractmethod**: 装饰器，强制子类实现该方法
- **类型提示**: `Optional[BaseChatModel | Embeddings]` 表示返回值可以是模型实例或 None

#### 工厂实现类

```python
class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        return ChatTongyi(model=rag_config["chat_model_name"])
```

- **ChatTongyi**: LangChain 封装的通义千问模型
- **model 参数**: 从配置文件读取模型名称

#### 全局实例

```python
chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()
```

- 模块加载时自动创建全局单例
- 其他模块可以直接导入使用

---

## 3. 配置管理

### 3.1 配置文件位置

`config/rag.yaml`:

```yaml
chat_model_name: "qwen3-max-2026-01-23"
embedding_model_name: "text-embedding-v4"
```

### 3.2 配置读取

`utils/config_handler.py`:

```python
import yaml

def load_config(config_name: str) -> dict:
    """加载 YAML 配置文件"""
    config_path = get_abs_path(f"config/{config_name}.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

rag_config = load_config("rag")
chroma_config = load_config("chroma")
agent_config = load_config("agent")
```

---

## 4. LangChain 模型接口

### 4.1 ChatModel 接口

LangChain 提供了统一的 ChatModel 接口，不同模型都遵循相同的调用方式：

```python
from langchain.schema import HumanMessage

# 调用聊天模型
response = chat_model.invoke([
    HumanMessage(content="你好")
])

# response 是 AIMessage 对象
print(response.content)  # 模型回复
```

### 4.2 Embeddings 接口

```python
# 生成文本嵌入
embeddings = embedding_model.embed_query("你好，世界")

# 批量生成
embeddings = embedding_model.embed_documents([
    "文档1内容",
    "文档2内容"
])
```

---

## 5. DashScope 阿里云服务

### 5.1 什么是 DashScope

DashScope 是阿里云提供的模型服务，提供了丰富的 LLM 和 Embedding 模型：

- **LLM**: 通义千问系列（qwen-turbo, qwen-max, qwen-long 等）
- **Embedding**: 文本嵌入系列（text-embedding-v1, v2, v4）

### 5.2 API Key 配置

使用 DashScope 需要配置 API Key：

```python
import os
os.environ["DASHSCOPE_API_KEY"] = "your-api-key"
```

通常放在 `.env` 文件中（注意添加到 `.gitignore`）：

```
DASHSCOPE_API_KEY=sk-xxxxxxx
```

---

## 6. 总结

本节课学习了：

1. **工厂模式**: 统一管理模型创建，便于切换和维护
2. **抽象基类**: 定义标准接口，强制实现
3. **LangChain 接口**: 统一的 ChatModel 和 Embeddings 接口
4. **DashScope 集成**: 阿里云模型服务的使用方法

---

## 代码练习

### 练习 1.1：添加新模型支持

在 `model/factory.py` 中添加一个新的工厂类，支持 OpenAI GPT 模型：

```python
class OpenAIChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        # 实现 OpenAI 模型创建
        pass
```

### 练习 1.2：模型切换功能

创建一个函数，根据环境变量切换使用不同模型：

```python
def get_chat_model():
    """根据环境变量返回不同的模型"""
    import os
    provider = os.environ.get("MODEL_PROVIDER", "dashscope")
    
    if provider == "openai":
        return OpenAIChatModelFactory().generator()
    else:
        return ChatModelFactory().generator()
```

### 练习 1.3：模型配置验证

在工厂类中添加配置验证，确保模型名称不为空：

```python
class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        model_name = rag_config.get("chat_model_name")
        if not model_name:
            raise ValueError("chat_model_name 未配置")
        return ChatTongyi(model=model_name)
```

---

## 下节课预告

下一课我们将学习 **向量数据库与 Embedding**，包括：
- Chroma 向量数据库的使用
- 文档加载与文本分块
- MD5 去重机制

---

## 相关资源

- [LangChain Chat Models 文档](https://python.langchain.com/docs/modules/model_io/chat/)
- [DashScope 官方文档](https://dashscope.aliyuncs.com/)
- [工厂模式详解](https://refactoringguru.cn/design-patterns/factory-method/python/en)
