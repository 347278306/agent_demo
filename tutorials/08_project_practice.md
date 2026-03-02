# 第八课：综合实战 - 从零构建智能问答系统

## 课程目标

1. 整合所有知识点
2. 完成一个完整的智能问答系统
3. 理解项目架构设计
4. 掌握开发调试技巧

---

## 1. 项目规划

### 1.1 功能需求

我们将要构建一个 **智能客服问答系统**，具备以下功能：

| 功能 | 描述 |
|------|------|
| 知识库问答 | 基于文档的智能问答 |
| 工具调用 | 支持外部工具扩展 |
| 多轮对话 | 支持上下文记忆 |
| Web 界面 | 友好的交互界面 |

### 1.2 技术架构

```
┌─────────────────────────────────────────────────────┐
│                   Web 层 (Streamlit)                │
│                   app.py                            │
└─────────────────────────┬───────────────────────────┘
                        │
┌─────────────────────────▼───────────────────────────┐
│                Agent 层 (LangChain)                 │
│              react_agent.py                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Tools    │  │ Middleware  │  │ System Prompt│ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────┬───────────────────────────┘
                        │
┌─────────────────────────▼───────────────────────────┐
│                  RAG 层                              │
│            rag_service.py                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Retriever  │  │  Prompt     │  │    LLM     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────┬───────────────────────────┘
                        │
┌─────────────────────────▼───────────────────────────┐
│                  基础层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Model     │  │  VectorDB   │  │  Config     │ │
│  │  Factory   │  │   (Chroma)  │  │   (YAML)    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 1.3 项目结构

```
my_chatbot/
├── app.py                    # Web 应用入口
├── model/
│   └── factory.py            # 模型工厂
├── rag/
│   ├── vector_store.py       # 向量存储
│   └── rag_service.py       # RAG 服务
├── agent/
│   ├── chat_agent.py         # Agent 核心
│   └── tools/
│       ├── __init__.py
│       ├── my_tools.py       # 自定义工具
│       └── middleware.py    # 中间件
├── config/
│   ├── model.yaml            # 模型配置
│   ├── chroma.yaml           # 向量库配置
│   └── prompts.yaml          # 提示词配置
├── prompts/
│   ├── system_prompt.txt     # 系统提示词
│   └── rag_prompt.txt       # RAG 提示词
├── data/
│   └── knowledge/             # 知识库文档
└── requirements.txt          # 依赖
```

---

## 2. 逐步实现

### 2.1 步骤一：配置管理

首先创建配置文件：

```yaml
# config/model.yaml
model_provider: "dashscope"
chat_model_name: "qwen3-max"
embedding_model_name: "text-embedding-v4"
```

```yaml
# config/chroma.yaml
collection_name: "my_knowledge"
persist_directory: "data/chroma_db"
chunk_size: 500
chunk_overlap: 50
k: 5
```

```yaml
# config/prompts.yaml
system_prompt_path: "prompts/system_prompt.txt"
rag_prompt_path: "prompts/rag_prompt.txt"
```

### 2.2 步骤二：模型工厂

```python
# model/factory.py

import os
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from utils.config_handler import model_config


class ChatModelFactory:
    def create(self) -> ChatTongyi:
        model_name = model_config.get("chat_model_name", "qwen3-max")
        return ChatTongyi(model=model_name)


class EmbeddingsFactory:
    def create(self) -> Embeddings:
        model_name = model_config.get("embedding_model_name", "text-embedding-v4")
        return DashScopeEmbeddings(model=model_name)


chat_model = ChatModelFactory().create()
embedding_model = EmbeddingsFactory().create()
```

### 2.3 步骤三：向量存储

```python
# rag/vector_store.py

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config_handler import chroma_config
from model.factory import embedding_model


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_config["persist_directory"],
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
        )
    
    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_config["k"]}
        )
    
    def add_documents(self, documents):
        self.vector_store.add_documents(documents)
    
    def load_knowledge(self, data_dir: str):
        """加载知识库文档"""
        from utils.file_handler import listdir_with_allowed_type
        
        files = listdir_with_allowed_type(
            data_dir, 
            tuple(chroma_config.get("allow_file_type", ["txt", "pdf"]))
        )
        
        for file_path in files:
            # 加载文档、分块、存储
            pass
```

### 2.4 步骤四：RAG 服务

```python
# rag/rag_service.py

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.factory import chat_model
from rag.vector_store import VectorStoreService


class RagService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        
        # 加载提示词模板
        with open("prompts/rag_prompt.txt", "r", encoding="utf-8") as f:
            template = f.read()
        
        self.prompt = PromptTemplate.from_template(template)
        self.chain = self.prompt | chat_model | StrOutputParser()
    
    def answer(self, question: str) -> str:
        # 1. 检索相关文档
        docs = self.retriever.invoke(question)
        
        # 2. 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. 生成回答
        return self.chain.invoke({"question": question, "context": context})
```

### 2.5 步骤五：自定义工具

```python
# agent/tools/my_tools.py

from langchain_core.tools import tool


@tool
def search_knowledgebase(query: str) -> str:
    """从知识库中搜索相关信息
    
    Args:
        query: 搜索关键词
        
    Returns:
        相关文档内容
    """
    from rag.rag_service import RagService
    rag = RagService()
    return rag.answer(query)


@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2+3*5"
        
    Returns:
        计算结果
    """
    try:
        # 注意：生产环境应该使用安全的计算方式
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"
```

### 2.6 步骤六：Agent 创建

```python
# agent/chat_agent.py

from langchain.agents import create_agent
from model.factory import chat_model
from agent.tools.my_tools import search_knowledgebase, get_current_time, calculator
from utils.prompt_loader import load_prompt


class ChatAgent:
    def __init__(self):
        system_prompt = load_prompt("prompts/system_prompt.txt")
        
        self.agent = create_agent(
            model=chat_model,
            system_prompt=system_prompt,
            tools=[
                search_knowledgebase,
                get_current_time,
                calculator,
            ]
        )
    
    def chat(self, message: str) -> str:
        response = self.agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })
        
        return response["messages"][-1].content
    
    def stream(self, message: str):
        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values"
        ):
            if chunk["messages"]:
                yield chunk["messages"][-1].content
```

### 2.7 步骤七：Web 应用

```python
# app.py

import streamlit as st
from agent.chat_agent import ChatAgent


def main():
    st.title("智能问答助手")
    st.divider()
    
    # 初始化
    if "agent" not in st.session_state:
        st.session_state["agent"] = ChatAgent()
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # 显示历史消息
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # 用户输入
    prompt = st.chat_input("请输入您的问题")
    
    if prompt:
        # 显示用户消息
        st.chat_message("user").write(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        # Agent 处理
        with st.spinner("思考中..."):
            response = st.session_state["agent"].stream(prompt)
            
            # 流式显示
            full_response = []
            response_box = st.chat_message("assistant")
            
            for chunk in response:
                if chunk:
                    full_response.append(chunk)
                    response_box.write(chunk, stream=True)
        
        # 保存消息
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "".join(full_response)
        })
        
        st.rerun()


if __name__ == "__main__":
    main()
```

---

## 3. 系统提示词设计

### 3.1 系统提示词

```txt
# prompts/system_prompt.txt

你是一个智能问答助手，可以通过工具与用户交互。

## 可用工具
1. search_knowledgebase: 搜索知识库
2. get_current_time: 获取当前时间
3. calculator: 计算数学表达式

## 工作流程
1. 理解用户问题
2. 判断是否需要调用工具
3. 如果需要，调用相应工具
4. 根据工具返回结果生成回答

## 回答要求
- 简洁明了
- 如果知识库无法回答，诚实用户
- 使用中文回复
```

### 3.2 RAG 提示词

```txt
# prompts/rag_prompt.txt

你是一个专业助手，请根据以下参考资料回答问题。

参考资料：
{context}

问题：{question}

要求：
1. 仅根据参考资料回答
2. 如资料不足，说明"根据现有资料无法回答"
3. 回答要准确、简洁
```

---

## 4. 运行与调试

### 4.1 启动应用

```bash
# 1. 激活虚拟环境
.venv\Scripts\activate

# 2. 加载知识库
python -c "from rag.vector_store import VectorStoreService; VectorStoreService().load_knowledge('data/knowledge')"

# 3. 启动 Web 应用
streamlit run app.py
```

### 4.2 调试技巧

```python
# 添加日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 在 Agent 中打印中间状态
def chat(self, message: str) -> str:
    print(f"用户输入: {message}")
    
    response = self.agent.invoke({...})
    
    print(f"Agent 输出: {response}")
    return response["messages"][-1].content
```

### 4.3 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 模型未配置 | API Key 未设置 | 设置 DASHSCOPE_API_KEY |
| 知识库为空 | 未加载文档 | 运行 load_knowledge |
| 工具调用失败 | 工具实现错误 | 检查工具返回值 |

---

## 5. 扩展功能

### 5.1 添加新工具

```python
# agent/tools/weather.py

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    # 实现天气查询逻辑
    pass

# 在 Agent 中注册
tools = [search_knowledgebase, get_current_time, calculator, get_weather]
```

### 5.2 添加中间件

```python
# agent/middleware/logging.py

@before_model
def log_request(state, runtime):
    print(f"消息历史: {len(state['messages'])} 条")
    return None

# 注册中间件
middleware = [log_request]
```

### 5.3 添加记忆功能

```python
class ChatAgent:
    def __init__(self):
        # 添加记忆
        from langchain.memory import ConversationBufferMemory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
```

---

## 6. 总结

本课程完成了：

1. **配置管理**: YAML 配置文件
2. **模型工厂**: 统一管理 LLM 和 Embedding
3. **向量存储**: Chroma 向量数据库
4. **RAG 服务**: 检索增强生成
5. **工具系统**: 自定义 Agent 工具
6. **中间件**: 日志、监控等功能
7. **Web 应用**: Streamlit 交互界面

---

## 后续学习建议

1. **深入 LangChain**: 学习更多 Chain 类型
2. **Agent 框架**: 探索 LangGraph、AutoGen
3. **向量数据库**: 学习 Pinecone、Milvus
4. **部署**: Docker、Kubernetes
5. **优化**: 提示词工程、性能调优

---

## 相关资源

- [LangChain 官方教程](https://python.langchain.com/docs/get_started/)
- [Streamlit 教程](https://docs.streamlit.io/library/get-started)
- [RAG 最佳实践](https://python.langchain.com/docs/use_cases/question_answering/)
- [Agent 设计模式](https://python.langchain.com/docs/modules/agents/)

---

## 恭喜完成！

经过八节课的学习，你已经掌握了：
- ✅ 大语言模型与 API 集成
- ✅ 向量数据库与文档处理
- ✅ RAG 检索增强生成
- ✅ AI Agent 与 ReAct 模式
- ✅ 工具设计与实现
- ✅ 中间件机制
- ✅ Web 应用开发
- ✅ 综合项目实战

继续探索 AI 开发的无限可能！
