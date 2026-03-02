# 第四课：AI Agent 基础与 ReAct 模式

## 课程目标

1. 理解 AI Agent 与普通 LLM 的区别
2. 掌握 LangChain Agent 的创建方法
3. 理解 ReAct 推理模式
4. 学会使用 create_agent 函数

---

## 1. AI Agent 概述

### 1.1 什么是 AI Agent

**Agent（智能体）** 是一个能够自主思考、规划和执行任务的系统。与普通 LLM 的区别：

| 特性 | 普通 LLM | AI Agent |
|------|----------|----------|
| 交互方式 | 被动响应 | 主动规划 |
| 工具使用 | 不可用 | 可调用工具 |
| 状态管理 | 无状态 | 有状态 |
| 复杂任务 | 一次性回答 | 逐步分解执行 |

### 1.2 Agent 的核心能力

```
Agent = LLM + 工具 + 记忆 + 规划
```

1. **LLM**: 理解和生成自然语言
2. **工具**: 执行特定操作（如搜索、计算）
3. **记忆**: 存储对话历史和中间结果
4. **规划**: 将复杂任务分解为步骤

### 1.3 Agent 工作流程

```
┌─────────────┐
│  用户输入   │
│ "帮我查天气"│
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  LLM 思考   │ ──▶ │ 需要调用工具 │
│ "要查深圳天气│     │ 调用get_weather│
└──────┬──────┘     └──────┬──────┘
       │                   │
       │                   ▼
       │            ┌─────────────┐
       │            │ 执行工具     │
       │            │ get_weather  │
       │            │ ("深圳")     │
       │            └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ 整合结果     │ ◀── │ 返回工具结果 │
│ "深圳今天    │     │ "晴天，28℃" │
│ 是晴天..."   │     │             │
└─────────────┘     └─────────────┘
```

---

## 2. LangChain Agent

### 2.1 create_agent 函数

LangChain 提供了 `create_agent` 函数来创建 Agent：

```python
from langchain.agents import create_agent

agent = create_agent(
    model=chat_model,              # LLM 模型
    system_prompt=system_prompt,  # 系统提示词
    tools=[tool1, tool2, ...],     # 可用工具列表
    middleware=[...]                # 中间件（可选）
)
```

### 2.2 项目源码分析

```python
# agent/react_agent.py

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    rag_summarize, get_weather, get_user_id,
    get_user_location, get_current_month,
    fill_context_for_report, fetch_external_data
)
from agent.tools.middleware import (
    monitor_tool, log_before_model, report_prompt_switch
)


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,           # RAG 问答工具
                get_weather,             # 天气查询
                get_user_id,             # 获取用户ID
                get_user_location,       # 获取用户位置
                get_current_month,       # 获取当前月份
                fill_context_for_report, # 报告生成上下文注入
                fetch_external_data,     # 外部数据查询
            ],
            middleware=[
                monitor_tool,           # 工具监控
                log_before_model,       # 模型调用前日志
                report_prompt_switch    # 动态提示词切换
            ]
        )
```

### 2.3 Agent 调用

```python
def execute_stream(self, query: str):
    # 构建输入
    input_dict = {
        "messages": [
            {"role": "user", "content": query}
        ]
    }
    
    # 流式执行
    for chunk in self.agent.stream(
        input_dict,
        stream_mode="values",
        context={"report": False}
    ):
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            yield latest_message.content.strip() + "\n"
```

---

## 3. ReAct 推理模式

### 3.1 什么是 ReAct

**ReAct**（Reasoning + Acting）是一种结合推理和行动的 Agent 架构：

- **Reason（推理）**: 分析问题，决定下一步行动
- **Act（行动）**: 执行工具调用或生成回答

```
用户问题 → 思考（Reason）→ 行动（Act）→ 观察结果 → 思考 → ... → 最终回答
```

### 3.2 ReAct 工作流程

以 "帮我查深圳今天的天气" 为例：

```
Step 1: 思考
   用户想查天气，需要调用 get_weather 工具，参数是"深圳"

Step 2: 行动
   调用 get_weather(city="深圳")

Step 3: 观察
   返回结果："晴天，28°C，湿度60%"

Step 4: 思考
   已获取天气信息，现在可以回答用户问题了

Step 5: 行动
   生成最终回答："深圳今天天气晴朗，气温28°C..."
```

### 3.3 ReAct vs 传统方式

| 方式 | 处理流程 | 问题 |
|------|----------|------|
| 传统 LLM | 直接回答 | 无法获取实时信息 |
| ReAct Agent | 思考→行动→观察→回答 | 可以调用工具获取实时信息 |

### 3.4 系统提示词设计

项目的系统提示词（`prompts/main_prompt.txt`）中定义了 ReAct 准则：

```
### 核心思考准则
1. 先判断用户的核心需求，分析「当前已有的信息（用户问题）是否足够直接回答」，若不足，思考需要调用什么工具获取缺失信息；
2. 调用工具获取结果后，**再次判断「工具返回的信息是否能完整、专业地回答用户问题」**：
   若信息足够：整合信息生成流畅、专业的最终回答；
   若信息不足/需要补充专业细节：自主判断需要再次调用哪个工具，明确工具入参，发起二次工具调用；
   若5次工具调用后仍信息不足，则回复用户：我不知道
```

---

## 4. Agent 状态管理

### 4.1 状态结构

LangChain Agent 使用 `AgentState` 管理状态：

```python
from langchain.agents import AgentState

state: AgentState = {
    "messages": [
        HumanMessage(content="帮我查天气"),
        AIMessage(content="思考过程..."),
        ToolMessage(content="晴天，28°C"),
        AIMessage(content="深圳今天天气..."),
    ],
    # 其他自定义状态
}
```

### 4.2 Context 上下文

在调用 Agent 时可以传入 context：

```python
for chunk in self.agent.stream(
    input_dict,
    stream_mode="values",
    context={"report": False}  # 自定义上下文
):
    # 处理每个 chunk
    pass
```

context 可以在中间件中访问：

```python
@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    # runtime.context 包含传入的 context
    report_mode = runtime.context.get("report", False)
```

---

## 5. 流式输出

### 5.1 stream 方法

LangChain Agent 支持流式输出：

```python
for chunk in agent.stream(input_dict, stream_mode="values"):
    # chunk 是每个步骤的结果
    pass
```

### 5.2 stream_mode 参数

| 模式 | 说明 |
|------|------|
| "values" | 返回每步的完整状态 |
| "messages" | 仅返回消息内容 |

### 5.3 项目中的实现

```python
def execute_stream(self, query: str):
    input_dict = {
        "messages": [
            {"role": "user", "content": query}
        ]
    }
    
    for chunk in self.agent.stream(
        input_dict,
        stream_mode="values",
        context={"report": False}
    ):
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            yield latest_message.content.strip() + "\n"
```

---

## 6. 总结

本节课学习了：

1. **AI Agent**: 具备工具调用能力的智能系统
2. **create_agent**: LangChain 创建 Agent 的函数
3. **ReAct 模式**: 推理 + 行动交替的架构
4. **状态管理**: AgentState 和 context
5. **流式输出**: stream 方法的使用

---

## 代码练习

### 练习 4.1：创建简单 Agent

使用 create_agent 创建一个简单的问答 Agent：

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    # 提示：使用 eval() 或 ast.literal_eval()
    pass

# 创建 Agent
agent = create_agent(
    model=chat_model,
    system_prompt="你是一个数学助手，可以使用计算器工具。",
    tools=[calculator]
)
```

### 练习 4.2：添加对话历史

扩展 Agent 支持多轮对话：

```python
class ChatAgent:
    def __init__(self):
        self.agent = create_agent(...)
        self.history = []  # 存储对话历史
    
    def chat(self, user_input: str):
        # 将历史消息添加到输入中
        messages = self.history + [HumanMessage(content=user_input)]
        # 调用 agent
        # 更新历史
        pass
```

### 练习 4.3：实现 ReAct 提示词

设计一个完整的 ReAct 系统提示词：

```python
REACT_PROMPT = """你是一个智能助手，采用 ReAct 模式工作。

## 工作流程
1. 思考：分析用户问题，判断是否需要调用工具
2. 行动：调用适当的工具获取信息
3. 观察：分析工具返回的结果
4. 回答：基于所有信息生成最终回答

## 输出格式
Thought: [你的思考过程]
Action: [工具名称]
Action Input: [工具参数]
Observation: [工具返回结果]
...（可重复）
Final Answer: [最终回答]
"""
```

---

## 下节课预告

下一课我们将学习 **工具（Tools）设计与实现**，包括：
- @tool 装饰器的使用
- 工具参数定义
- 工具返回值处理

---

## 相关资源

- [LangChain Agents 文档](https://python.langchain.com/docs/modules/agents/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [LangChain create_agent API](https://api.python.langchain.com/en/latest/agents/langchain.agents.create_agent.html)
