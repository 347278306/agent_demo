# 第七课：Streamlit Web 应用开发

## 课程目标

1. 掌握 Streamlit 基础组件
2. 理解 Session State 状态管理
3. 学会实现流式输出
4. 理解 ReactAgent 与 Web 应用的集成

---

## 1. Streamlit 概述

### 1.1 什么是 Streamlit

Streamlit 是一个用 Python 创建 Web 应用的框架，无需 HTML/CSS/JS：

```python
import streamlit as st

st.title("我的应用")
st.write("Hello, World!")
```

### 1.2 项目应用

```
app.py → ReactAgent → 流式响应 → Web 界面
```

---

## 2. 源码分析

### 2.1 完整代码

```python
# app.py

import time
import streamlit as st
from agent.react_agent import ReactAgent

# 页面配置
st.title("智能客服")
st.divider()

# 初始化 Session State
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 显示历史消息
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 用户输入
prompt = st.chat_input()

if prompt:
    # 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # Agent 处理
    response_messages = []
    with st.spinner("智能客服思考中..."):
        # 流式获取响应
        res_stream = st.session_state["agent"].execute_stream(prompt)
        
        # 定义捕获函数
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)  # 打字机效果
                    yield char
        
        # 显示助手消息
        st.chat_message("assistant").write(
            capture(res_stream, response_messages)
        )
        
        # 保存完整消息
        st.session_state["messages"].append({
            "role": "assistant", 
            "content": response_messages[-1]
        })
        
        st.rerun()
```

---

## 3. 核心组件

### 3.1 页面标题

```python
st.title("智能客服")           # 大标题
st.header("子标题")            # 子标题  
st.subheader("小标题")         # 小标题
st.markdown("**加粗文本**")    # Markdown 支持
```

### 3.2 分割线

```python
st.divider()   # 水平分割线
```

### 3.3 聊天组件

```python
# 显示聊天消息
st.chat_message("user").write("用户消息")
st.chat_message("assistant").write("助手消息")
st.chat_message("assistant").write(message, is_template=True)

# 用户输入框
prompt = st.chat_input("请输入您的问题")
```

### 3.4 加载状态

```python
# 旋转加载动画
with st.spinner("处理中..."):
    result = do_something()
# spinner 自动消失
```

---

## 4. Session State

### 4.1 什么是 Session State

Session State 是 Streamlit 的状态管理机制，用于在页面刷新间保持数据：

```python
# 初始化
if "key" not in st.session_state:
    st.session_state["key"] = value

# 使用
st.session_state["key"]

# 修改
st.session_state["key"] = new_value
```

### 4.2 项目中的应用

```python
# 1. 存储 Agent 实例（避免重复创建）
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# 2. 存储聊天历史
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 3. 追加新消息
st.session_state["messages"].append({"role": "user", "content": prompt})
```

### 4.3 状态生命周期

```
用户访问 → 创建 Session → 页面交互 → 刷新/关闭 → Session 结束
                    ↓
            session_state 保持数据
```

### 4.4 回调函数

```python
def increment_counter():
    st.session_state.counter += 1

st.button("点击", on_click=increment_counter)
st.write(f"计数: {st.session_state.counter}")
```

---

## 5. 流式输出

### 5.1 基本流式输出

```python
# Agent 返回生成器
def execute_stream(query):
    for chunk in agent.stream(query):
        yield chunk

# Streamlit 中使用
for char in execute_stream("你好"):
    # 逐字显示
    st.write(char)
```

### 5.2 项目中的实现

```python
def capture(generator, cache_list):
    """捕获流式输出的同时实现打字机效果"""
    for chunk in generator:
        cache_list.append(chunk)  # 缓存完整响应
        
        for char in chunk:        # 逐字符输出
            time.sleep(0.01)
            yield char

# 使用
res_stream = agent.execute_stream(prompt)
st.chat_message("assistant").write(capture(res_stream, response_messages))
```

### 5.3 流式输出原理

```
原始输出：["你好", "，", "我", "是", "Agent"]
            ↓
字符流：   "你" → "好" → "，" → "我" → "是" → "A" → "g" → "e" → "n" → "t"
            ↓
显示效果：打字机逐字显示
```

### 5.4 完整响应捕获

```python
response_messages = []

def capture(generator, cache_list):
    for chunk in generator:
        cache_list.append(chunk)
        for char in chunk:
            time.sleep(0.01)
            yield char

# 执行后，response_messages 包含完整响应
st.chat_message("assistant").write(capture(res_stream, response_messages))

# 可以访问完整响应
full_response = response_messages[-1]
```

---

## 6. 页面刷新

### 6.1 st.rerun()

```python
st.rerun()   # 重新运行整个脚本
```

作用：
- 触发页面重新渲染
- 更新 session_state
- 刷新界面

### 6.2 项目中的应用

```python
# 保存消息后刷新页面，显示新消息
st.session_state["messages"].append({...})
st.rerun()
```

### 6.3 刷新 vs 不刷新

| 场景 | 方法 |
|------|------|
| 更新 UI 显示 | st.rerun() |
| 保持当前状态 | 不调用 rerun() |
| 条件刷新 | if condition: st.rerun() |

---

## 7. 完整对话流程

### 7.1 时序图

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 用户输入    │ ──▶ │ Agent 处理  │ ──▶ │ 流式显示   │
│ "扫地机器人 │     │ execute_    │     │ capture     │
│ 电池多久？" │     │ stream()    │     │ 逐字显示    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ 保存消息    │
                                        │ rerun 刷新  │
                                        └─────────────┘
```

### 7.2 核心代码

```python
# 1. 用户输入
prompt = st.chat_input()

if prompt:
    # 2. 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # 3. Agent 流式处理
    with st.spinner("智能客服思考中..."):
        response_messages = []
        res_stream = st.session_state["agent"].execute_stream(prompt)
        
        # 4. 流式显示
        st.chat_message("assistant").write(
            capture(res_stream, response_messages)
        )
    
    # 5. 保存并刷新
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response_messages[-1]
    })
    st.rerun()
```

---

## 8. 高级功能

### 8.1 侧边栏

```python
st.sidebar.title("设置")
model = st.sidebar.selectbox("选择模型", ["qwen-max", "qwen-turbo"])
temperature = st.sidebar.slider("温度", 0.0, 1.0, 0.7)
```

### 8.2 多列布局

```python
col1, col2 = st.columns(2)

with col1:
    st.write("第一列")

with col2:
    st.write("第二列")
```

### 8.3 标签页

```python
tab1, tab2, tab3 = st.tabs(["问答", "知识库", "设置"])

with tab1:
    st.write("问答界面")

with tab2:
    st.write("知识库管理")

with tab3:
    st.write("系统设置")
```

---

## 9. 总结

本节课学习了：

1. **Streamlit 基础**: 页面组件、聊天组件、输入输出
2. **Session State**: 状态保持与数据共享
3. **流式输出**: 打字机效果实现
4. **页面刷新**: rerun 的使用场景

---

## 代码练习

### 练习 7.1：添加清空聊天功能

在 app.py 中添加清空聊天按钮：

```python
if st.button("清空聊天"):
    st.session_state["messages"] = []
    st.rerun()
```

### 练习 7.2：添加聊天历史记录显示

显示完整的对话历史：

```python
# 在页面侧边栏显示历史记录
with st.sidebar:
    st.write("## 对话历史")
    for msg in st.session_state["messages"]:
        st.write(f"**{msg['role']}**: {msg['content'][:50]}...")
```

### 练习 7.3：添加模型选择器

在侧边栏添加模型选择：

```python
# 侧边栏
with st.sidebar:
    selected_model = st.selectbox(
        "选择模型",
        ["qwen3-max", "qwen3-turbo"],
        index=0
    )
    
    # 根据选择创建不同的 Agent
    if "current_model" not in st.session_state or st.session_state["current_model"] != selected_model:
        st.session_state["agent"] = ReactAgent(model_name=selected_model)
        st.session_state["current_model"] = selected_model
```

### 练习 7.4：添加打字速度控制

添加滑块控制打字速度：

```python
typing_speed = st.sidebar.slider("打字速度", 0, 100, 50)

def capture_with_speed(generator, cache_list, speed):
    delay = (100 - speed) / 1000  # 转换为秒
    for chunk in generator:
        cache_list.append(chunk)
        for char in chunk:
            time.sleep(delay)
            yield char
```

---

## 下节课预告

下一课我们将学习 **综合实战：从零构建问答系统**，整合所有知识点完成一个完整的项目。

---

## 相关资源

- [Streamlit 官方文档](https://docs.streamlit.io/)
- [Streamlit Chat 组件](https://docs.streamlit.io/library/api-reference/chat)
- [Session State 指南](https://docs.streamlit.io/library/api-reference/state)
