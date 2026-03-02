# 第三课：RAG 服务与链式调用

## 课程目标

1. 理解 RAG 的完整工作流程
2. 掌握 LangChain Chain 链式调用
3. 学会使用 PromptTemplate
4. 理解 OutputParser 输出解析

---

## 1. RAG 概述

### 1.1 什么是 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的技术：

```
用户问题 → 检索相关文档 → 将文档作为上下文 → LLM 生成答案
```

### 1.2 RAG 工作流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  用户问题   │ ──▶ │  向量检索   │ ──▶ │  构建提示词 │ ──▶ │ LLM 生成   │
│ "扫地机器人 │     │  找到相关   │     │  拼接上下文  │     │  最终答案  │
│  电池多久?" │     │  参考资料   │     │  "根据资料" │     │  "一般是"  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### 1.3 为什么需要 RAG

纯 LLM 的问题：
1. **知识截止**: 训练数据有时间限制，无法回答新问题
2. **幻觉**: LLM 可能生成看似合理但错误的内容
3. **专业领域**: 缺乏特定领域的专业知识

RAG 的优势：
1. **实时知识**: 可以检索最新文档
2. **可追溯**: 答案有据可查
3. **专业领域**: 可以注入专业知识库

---

## 2. 源码分析

### 2.1 RagSummarizeService 类

```python
class RagSummarizeService(object):
    def __init__(self):
        # 1. 初始化向量存储服务
        self.vector_store = VectorStoreService()
        
        # 2. 获取检索器
        self.retriever = self.vector_store.get_retriever()
        
        # 3. 加载提示词模板
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        
        # 4. 获取模型
        self.model = chat_model
        
        # 5. 初始化链
        self.chain = self._init_chain()
```

### 2.2 Chain 初始化

```python
def _init_chain(self):
    chain = (
        self.prompt_template 
        | print_prompt 
        | self.model 
        | StrOutputParser()
    )
    return chain
```

这就是 LangChain 的**管道操作符** `|`，它将多个组件串联起来。

---

## 3. LangChain Chain 详解

### 3.1 什么是 Chain

Chain（链）是 LangChain 的核心概念，它将多个组件串联起来：

```python
chain = prompt_template | model | output_parser
```

数据流向：
```
输入 → prompt_template → model → output_parser → 输出
  ↓          ↓            ↓           ↓
  "问题"   格式化提示词   LLM调用    解析输出
```

### 3.2 管道操作符原理

```python
# 等价于：
chain = RunnableSequence(
    first=prompt_template,
    middle=[model],
    last=StrOutputParser()
)

# 调用链
result = chain.invoke({"question": "扫地机器人电池多久？"})
```

### 3.3 Chain 的类型

LangChain 提供了多种内置 Chain：

| Chain | 用途 |
|-------|------|
| LLMChain | 基础链：Prompt + LLM |
| RetrievalQA | RAG 问答链 |
| ConversationChain | 对话链 |
| SequentialChain | 顺序链 |

---

## 4. PromptTemplate

### 4.1 模板语法

```python
from langchain_core.prompts import PromptTemplate

template = "请根据以下上下文回答问题。\n\n上下文：{context}\n\n问题：{input}"

prompt = PromptTemplate.from_template(template)
```

### 4.2 模板变量

```python
# 输入变量
prompt = PromptTemplate(
    template="请介绍{topic}。",
    input_variables=["topic"]
)

# 调用
prompt.invoke({"topic": "扫地机器人"})
```

### 4.3 项目中的模板

`prompts/rag_summarize.txt`:

```
你是一个专业的智能客服，请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{input}

请给出专业、详细的回答。如果参考资料无法回答问题，请如实说明。
```

### 4.4 模板解析过程

```python
prompt_text = load_rag_prompts()  # 加载模板文本

prompt_template = PromptTemplate.from_template(prompt_text)

# 实际调用时
formatted = prompt_template.invoke({
    "context": "【参考资料1】：扫地机器人电池...",
    "input": "电池能用多久？"
})

print(formatted.to_string())
# 输出：
# Human: 你是一个专业的智能客服...
# 
# 参考资料：【参考资料1】：扫地机器人电池...
# 
# 用户问题：电池能用多久？
```

---

## 5. Retriever 检索器

### 5.1 检索流程

```python
def retriever_docs(self, query: str) -> list[Document]:
    return self.retriever.invoke(query)
```

### 5.2 检索结果处理

```python
def rag_summarize(self, query: str) -> str:
    # 1. 检索相关文档
    context_docs = self.retriever_docs(query)
    
    # 2. 构建上下文字符串
    context = ""
    counter = 0
    for doc in context_docs:
        counter += 1
        context += f"【参考资料{counter}】：{doc.page_content} | 参考元数据：{doc.metadata}\n"
    
    # 3. 调用链生成答案
    return self.chain.invoke({
        "input": query,
        "context": context
    })
```

### 5.3 检索结果示例

```python
# 输入：query = "扫地机器人电池能用多久？"

# 检索结果：
[
    Document(
        page_content="扫地机器人电池续航时间一般为2-4小时...",
        metadata={"source": "data/扫地机器人100问.txt"}
    ),
    Document(
        page_content="不同型号的电池容量不同...",
        metadata={"source": "data/扫地机器人选购指南.txt"}
    ),
]

# 构建的 context：
"【参考资料1】：扫地机器人电池续航时间一般为2-4小时... | 参考元数据：{'source': 'data/扫地机器人100问.txt'}
【参考资料2】：不同型号的电池容量不同... | 参考元数据：{'source': 'data/扫地机器人选购指南.txt'}"
```

---

## 6. OutputParser

### 6.1 什么是 OutputParser

OutputParser 负责将 LLM 的输出解析成结构化数据：

```
LLM 输出 → OutputParser → 结构化结果
"扫地机器人电池   →   {"answer": "扫地机器人电池
 能用2-4小时"                  "能用2-4小时"}
```

### 6.2 常用 Parser

```python
from langchain_core.output_parsers import StrOutputParser

# 字符串输出（最简单）
parser = StrOutputParser()

# JSON 输出
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()

# Pydantic 输出
from langchain_core.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Answer)
```

### 6.3 项目中的使用

```python
chain = (
    self.prompt_template 
    | print_prompt       # 打印提示词（调试用）
    | self.model         # 调用 LLM
    | StrOutputParser()  # 解析输出为字符串
)

result = chain.invoke({"input": query, "context": context})
# result 是字符串类型的回答
```

---

## 7. 完整调用流程

### 7.1 时序图

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│ 用户调用       │     │ RagSummarize   │     │ Chain 执行    │
│ rag_summarize │ ──▶ │ Service        │ ──▶ │                │
│ ("电池能用    │     │                │     │                │
│  多久？")      │     │                │     │                │
└────────────────┘     └────────────────┘     └────────────────┘
                                                    │
                       ┌────────────────┐           │
                       │ VectorStore    │ ◀──────────┤
                       │ Service        │           │
                       │                │           ▼
                       └────────────────┘     ┌────────────────┐
                                             │ Retriever      │
                                             │ 检索相关文档   │
                                             └────────────────┘
                                                    │
                       ┌────────────────┐           │
                       │ prompt_loader  │ ◀──────────┤
                       │ 加载提示词模板 │           ▼
                       └────────────────┘     ┌────────────────┐
                                             │ PromptTemplate │
                                             │ 构建完整提示词 │
                                             └────────────────┘
                                                    │
                       ┌────────────────┐           │
                       │ model/factory  │ ◀──────────┤
                       │ chat_model     │           ▼
                       └────────────────┘     ┌────────────────┐
                                             │ LLM            │
                                             │ 生成答案       │
                                             └────────────────┘
                                                    │
                       ┌────────────────┐           │
                       │ StrOutputParser│ ◀──────────┤
                       │ 解析输出       │           ▼
                       └────────────────┘     ┌────────────────┐
                                             │ 返回最终答案   │
                                             └────────────────┘
```

### 7.2 关键代码

```python
class RagSummarizeService:
    def rag_summarize(self, query: str) -> str:
        # Step 1: 检索相关文档
        context_docs = self.retriever.invoke(query)
        
        # Step 2: 构建上下文
        context = self._build_context(context_docs)
        
        # Step 3: 调用链生成答案
        result = self.chain.invoke({
            "input": query,
            "context": context
        })
        
        return result
    
    def _build_context(self, docs: list[Document]) -> str:
        context = ""
        for i, doc in enumerate(docs, 1):
            context += f"【参考资料{i}】：{doc.page_content}\n"
        return context
```

---

## 8. 总结

本节课学习了：

1. **RAG 原理**: 检索 + 生成的组合
2. **Chain 链式调用**: 用管道操作符串联组件
3. **PromptTemplate**: 提示词模板的使用
4. **Retriever**: 向量检索器
5. **OutputParser**: 输出解析器

---

## 代码练习

### 练习 3.1：自定义输出格式

修改 Chain，使用 JSONOutputParser 返回结构化答案：

```python
from langchain_core.output_parsers import JsonOutputParser

class Answer(BaseModel):
    answer: str
    sources: list[str]

# 创建 parser 和 chain
# 提示：需要在 prompt 中告诉 LLM 输出 JSON 格式
```

### 练习 3.2：多轮对话 RAG

扩展 RagSummarizeService 支持多轮对话上下文：

```python
class RagSummarizeService:
    def __init__(self):
        # 添加对话历史
        self.conversation_history = []
    
    def chat(self, query: str) -> str:
        # 将历史对话和当前问题一起发送给 LLM
        pass
```

### 练习 3.3：混合检索

实现关键词 + 向量的混合检索：

```python
def get_hybrid_retriever(self):
    # 结合 BM25 关键词检索和向量检索
    # 提示：使用 LangChain 的 EnsembleRetriever
    pass
```

---

## 下节课预告

下一课我们将学习 **Agent 基础**，包括：
- 什么是 AI Agent
- create_agent 函数用法
- ReAct 推理模式

---

## 相关资源

- [LangChain Chains 文档](https://python.langchain.com/docs/modules/chains/)
- [RAG 架构详解](https://python.langchain.com/docs/use_cases/question_answering/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
