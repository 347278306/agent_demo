# 第六课：中间件（Middleware）机制

## 课程目标

1. 理解中间件的概念和作用
2. 掌握 @wrap_tool_call 工具拦截
3. 掌握 @before_model 模型调用前拦截
4. 掌握 @dynamic_prompt 动态提示词
5. 理解上下文状态管理

---

## 1. 中间件概述

### 1.1 什么是中间件

中间件是一种**拦截器模式**的实现，允许在请求处理过程中插入自定义逻辑：

```
请求 → 中间件1 → 中间件2 → 核心逻辑 → 中间件3 → 响应
         ↓           ↓                 ↓
      预处理1     预处理2            后处理
```

### 1.2 LangChain Agent 中间件

在 LangChain Agent 中，中间件可以拦截：

1. **工具调用前/后**: @wrap_tool_call
2. **模型调用前**: @before_model
3. **生成提示词前**: @dynamic_prompt

### 1.3 项目中的中间件架构

```python
# agent/react_agent.py

agent = create_agent(
    model=chat_model,
    system_prompt=load_system_prompts(),
    tools=[...],
    middleware=[
        monitor_tool,           # 工具监控
        log_before_model,       # 模型调用前日志
        report_prompt_switch    # 动态提示词切换
    ]
)
```

---

## 2. 源码分析

### 2.1 完整中间件代码

```python
# agent/tools/middleware.py

from langchain.agents import AgentState
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import (
    wrap_tool_call, before_model, dynamic_prompt, ModelRequest
)
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing import Callable
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts
```

### 2.2 中间件装饰器

LangChain 提供了多种中间件装饰器：

| 装饰器 | 作用 | 执行时机 |
|--------|------|----------|
| @wrap_tool_call | 拦截工具调用 | 工具执行前/后 |
| @before_model | 模型调用前 | LLM 调用前 |
| @dynamic_prompt | 动态生成提示词 | 生成提示词前 |

---

## 3. 工具拦截器 @wrap_tool_call

### 3.1 基本用法

```python
@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,        # 工具调用请求
    handler: Callable               # 实际处理函数
) -> ToolMessage | Command:
    # 预处理
    logger.info(f"执行工具：{request.tool_call['name']}")
    
    # 调用实际工具
    result = handler(request)
    
    # 后处理
    logger.info(f"工具执行完成")
    
    return result
```

### 3.2 项目实例：monitor_tool

```python
@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    # 1. 记录工具名称和参数
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")
    
    try:
        # 2. 执行工具
        result = handler(request)
        
        # 3. 特殊处理：检测报告生成场景
        if request.tool_call['name'] == "fill_context_for_report":
            # 修改运行时上下文
            request.runtime.context["report"] = True
        
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        
        return result
    
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e
```

### 3.3 ToolCallRequest 结构

```python
request: ToolCallRequest = {
    "tool_call": {
        "name": "get_weather",      # 工具名称
        "args": {"city": "深圳"},   # 工具参数
        "id": "call_xxx"            # 调用ID
    },
    "runtime": Runtime(...)         # 运行时上下文
}
```

### 3.4 Runtime 上下文

```python
# runtime.context 是可以跨调用共享的字典
request.runtime.context["report"] = True  # 设置标记
report_mode = runtime.context.get("report", False)  # 读取
```

---

## 4. 模型拦截器 @before_model

### 4.1 基本用法

```python
@before_model
def log_before_model(
    state: AgentState,   # Agent 状态
    runtime: Runtime    # 运行时上下文
):
    # 预处理逻辑
    logger.info(f"消息数量：{len(state['messages'])}")
    
    # 返回值：None 或 Command
    return None  # 继续正常流程
    # return Command(...)  # 跳过模型，直接返回
```

### 4.2 项目实例：log_before_model

```python
@before_model
def log_before_model(
    state: AgentState,   # 整个 Agent 中的状态记录
    runtime: Runtime,    # 记录了执行过程中的上下文信息
):
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条数据")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    
    return None  # 不拦截，继续正常流程
```

### 4.3 AgentState 结构

```python
state: AgentState = {
    "messages": [
        HumanMessage(content="帮我查天气"),      # 用户消息
        AIMessage(content="Thought: ..."),       # AI 思考
        ToolMessage(content="晴天"),             # 工具结果
        AIMessage(content="深圳今天天气..."),     # 最终回答
    ]
}
```

### 4.4 实际应用场景

```python
@before_model
def check_rate_limit(state: AgentState, runtime: Runtime):
    """检查请求频率限制"""
    import time
    
    current_time = time.time()
    last_request = runtime.context.get("last_request_time", 0)
    
    if current_time - last_request < 1:  # 1秒内不重复调用
        return Command(
            update={"messages": [...]}  # 跳过模型调用
        )
    
    runtime.context["last_request_time"] = current_time
    return None
```

---

## 5. 动态提示词 @dynamic_prompt

### 5.1 基本用法

```python
@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """根据上下文动态切换提示词"""
    
    if request.runtime.context.get("report", False):
        return load_report_prompts()   # 报告场景提示词
    else:
        return load_system_prompts()   # 普通场景提示词
```

### 5.2 项目实例：report_prompt_switch

```python
@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """动态切换提示词
    
    根据 context 中的 report 标记决定使用哪个提示词
    """
    if request.runtime.context.get("report", False):
        # 报告生成场景
        return load_report_prompts()
    else:
        # 普通问答场景
        return load_system_prompts()
```

### 5.3 完整工作流程

```
1. 用户说"帮我生成报告"
2. Agent 决定调用 fill_context_for_report
3. monitor_tool 拦截调用，设置 runtime.context["report"] = True
4. 再次调用模型前，report_prompt_switch 检测到 report=True
5. 动态切换到报告生成专用提示词
6. Agent 使用报告提示词继续执行
```

### 5.4 提示词切换效果

```python
# 普通提示词（prompts/main_prompt.txt）
"""
你是一个扫地机器人的专业智能客服。
回答用户关于产品使用、选购、维护等问题。
"""

# 报告提示词（prompts/report_prompt.txt）
"""
你是一个数据分析助手。
根据用户的使用数据，生成专业的分析报告。
包含：使用效率分析、耗材使用情况、改进建议等。
"""
```

---

## 6. 中间件组合使用

### 6.1 注册中间件

```python
agent = create_agent(
    model=chat_model,
    system_prompt=load_system_prompts(),
    tools=[...],
    middleware=[
        monitor_tool,           # 1. 工具监控
        log_before_model,       # 2. 日志记录
        report_prompt_switch    # 3. 提示词切换
    ]
)
```

### 6.2 中间件执行顺序

```
工具调用流程：
  monitor_tool (wrap_tool_call)
    ↓
  实际工具执行
    ↓
  monitor_tool 返回

模型调用流程：
  log_before_model (before_model)
    ↓
  report_prompt_switch (dynamic_prompt)
    ↓
  模型生成
```

---

## 7. 高级应用

### 7.1 条件跳过模型调用

```python
@before_model
def cache_check(state: AgentState, runtime: Runtime):
    """检查缓存"""
    last_message = state["messages"][-1]
    
    # 如果用户只是确认，直接返回
    if last_message.content in ["是的", "确认", "好"]:
        return Command(
            update={
                "messages": [
                    AIMessage(content="好的，我会继续执行。")
                ]
            }
        )
    
    return None
```

### 7.2 修改工具参数

```python
@wrap_tool_call
def parameter_validator(
    request: ToolCallRequest,
    handler: Callable
) -> ToolMessage | Command:
    # 验证参数
    args = request.tool_call['args']
    
    if "city" in args:
        # 规范化城市名称
        args["city"] = args["city"].strip()
    
    return handler(request)
```

### 7.3 性能监控

```python
import time

@wrap_tool_call
def performance_monitor(
    request: ToolCallRequest,
    handler: Callable
) -> ToolMessage | Command:
    start_time = time.time()
    
    result = handler(request)
    
    elapsed = time.time() - start_time
    logger.info(f"工具 {request.tool_call['name']} 执行耗时: {elapsed:.2f}s")
    
    return result
```

---

## 8. 总结

本节课学习了：

1. **中间件概念**: 拦截器模式的应用
2. **@wrap_tool_call**: 工具调用拦截，可修改参数、捕获异常
3. **@before_model**: 模型调用前拦截，可跳过调用
4. **@dynamic_prompt**: 动态生成提示词
5. **Runtime Context**: 跨步骤共享状态

---

## 代码练习

### 练习 6.1：重试中间件

创建一个工具重试中间件：

```python
@wrap_tool_call
def retry_tool(
    request: ToolCallRequest,
    handler: Callable,
    max_retries: int = 3
) -> ToolMessage | Command:
    """工具失败自动重试"""
    # 实现逻辑：失败时自动重试最多 max_retries 次
    pass
```

### 练习 6.2：限流中间件

创建限流中间件，防止频繁调用：

```python
@before_model
def rate_limiter(state: AgentState, runtime: Runtime):
    """每分钟最多调用模型 20 次"""
    # 使用 runtime.context 记录调用次数和时间
    pass
```

### 练习 6.3：敏感信息过滤

创建提示词过滤中间件：

```python
@dynamic_prompt
def sensitive_filter(request: ModelRequest):
    """过滤敏感信息"""
    # 检查是否包含敏感词，如有则替换
    pass
```

### 练习 6.4：多模态提示词切换

根据用户意图切换不同提示词：

```python
@dynamic_prompt
def intent_based_prompt(request: ModelRequest):
    """根据用户意图选择提示词"""
    # 分析用户问题类型
    # 切换到对应的专家提示词
    pass
```

---

## 下节课预告

下一课我们将学习 **Streamlit Web 应用开发**，包括：
- Streamlit 基础组件
- Session State 状态管理
- 流式输出实现

---

## 相关资源

- [LangChain Agent Middleware](https://python.langchain.com/docs/modules/agents/agent_concepts/middleware/)
- [LangGraph Middleware](https://langchain-ai.github.io/langgraphjs/guides/middleware/)
- [Middleware Pattern](https://refactoringguru.cn/design-patterns/decorator/python/example)
