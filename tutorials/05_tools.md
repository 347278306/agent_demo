# 第五课：工具（Tools）设计与实现

## 课程目标

1. 掌握 @tool 装饰器的使用
2. 理解工具参数的定义方式
3. 学会设计不同类型的工具
4. 理解工具在 Agent 中的作用

---

## 1. 工具概述

### 1.1 什么是工具

工具（Tools）是 Agent 与外部世界交互的桥梁。通过工具，Agent 可以：

- 搜索信息
- 执行计算
- 访问数据库
- 调用外部 API
- 操作文件系统

### 1.2 工具在 Agent 中的作用

```
用户问题 → Agent 判断 → 需要工具？ → 调用工具 → 获取结果 → 生成回答
                              ↓
                           不需要 → 直接生成回答
```

---

## 2. @tool 装饰器

### 2.1 基本用法

使用 LangChain 的 `@tool` 装饰器定义工具：

```python
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    # 工具实现
    return f"{city} 今天是晴天"
```

### 2.2 装饰器参数

```python
@tool(description="工具功能描述")
def tool_name(param: str) -> str:
    """工具的详细说明（可选）"""
    return result
```

- **description**: 工具描述，供 LLM 理解工具用途
- **函数名**: 自动成为工具名称
- **函数参数**: LLM 需要提供的参数
- **返回类型**: 工具的返回值描述

### 2.3 项目中的工具示例

```python
# agent/tools/agent_tools.py

@tool(description="获取指定城市的天气")
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return random.choice(["晴天", "阴天", "多云", "小雨", "中雨", "大雨", "小雪"])


@tool(description="获取用户所在城市名称")
def get_user_location() -> str:
    """无入参工具示例"""
    return random.choice(["深圳", "合肥", "杭州", "北京"])
```

---

## 3. 工具参数设计

### 3.1 带参数工具

```python
@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """检索知识库中的相关资料"""
    return rag.rag_summarize(query)
```

参数说明：
- `query: str`: 检索关键词
- 返回 `str`: 检索结果

### 3.2 无参数工具

```python
@tool(description="获取用户id")
def get_user_id() -> str:
    """获取当前用户ID"""
    return random.choice(user_ids)
```

无参数工具特点：
- 函数签名无参数
- LLM 直接调用，无需提供参数

### 3.3 多参数工具

```python
@tool(description="从外部系统中获取指定用户指定月份的使用记录")
def fetch_external_data(user_id: str, month: str) -> str:
    """获取用户指定月份的数据"""
    generate_external_data()
    
    try:
        return external_data[user_id][month]
    except KeyError:
        return ""
```

### 3.4 参数类型注解

LangChain 会根据类型注解生成工具 schema：

```python
@tool
def search_products(
    category: str,        # 必填参数
    min_price: float = 0,  # 可选参数，有默认值
    max_price: float = None  # 可选参数
) -> list[dict]:
    """搜索产品"""
    pass
```

生成的 schema：
```json
{
  "name": "search_products",
  "description": "搜索产品",
  "parameters": {
    "type": "object",
    "properties": {
      "category": {"type": "string"},
      "min_price": {"type": "number", "default": 0},
      "max_price": {"type": "number"}
    },
    "required": ["category"]
  }
}
```

---

## 4. 项目中的完整工具实现

### 4.1 RAG 工具

```python
from rag.rag_service import RagSummarizeService

rag = RagSummarizeService()


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """检索知识库中的相关资料
    
    Args:
        query: 检索关键词
        
    Returns:
        相关参考资料内容
    """
    return rag.rag_summarize(query)
```

### 4.2 外部数据工具

```python
import os
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path


external_data = {}


def generate_external_data():
    """从 CSV 文件加载外部数据"""
    global external_data
    
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])
        
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")
        
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:  # 跳过表头
                arr = line.strip().split(",")
                
                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time = arr[5].replace('"', "")
                
                if user_id not in external_data:
                    external_data[user_id] = {}
                
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户指定月份的使用记录")
def fetch_external_data(user_id: str, month: str) -> str:
    """获取用户指定月份的使用记录
    
    Args:
        user_id: 用户ID
        month: 月份，格式为 YYYY-MM
        
    Returns:
        用户使用记录，包含特征、效率、耗材、对比等信息
    """
    generate_external_data()
    
    try:
        return str(external_data[user_id][month])
    except KeyError:
        logger.warning(f"未能检索到用户 {user_id} 在 {month} 的使用记录")
        return ""
```

### 4.3 上下文注入工具

```python
@tool(description="无入参无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息")
def fill_context_for_report():
    """触发上下文注入，用于报告生成场景"""
    return "fill_context_for_report已调用"
```

这个工具的特殊之处：
- 无参数
- 返回值不重要
- 作用是触发中间件修改 Agent 行为

---

## 5. 工具设计模式

### 5.1 工具注册

在 Agent 创建时注册工具：

```python
from langchain.agents import create_agent

agent = create_agent(
    model=chat_model,
    system_prompt=system_prompt,
    tools=[
        rag_summarize,           # 注册 RAG 工具
        get_weather,             # 注册天气工具
        fetch_external_data,     # 注册外部数据工具
        # ... 更多工具
    ]
)
```

### 5.2 工具选择机制

LLM 会根据以下信息选择工具：

1. **工具描述**: `description` 参数
2. **工具名称**: 函数名
3. **参数类型**: 类型注解
4. **系统提示词**: Agent 系统提示中的工具说明

### 5.3 工具优先级

在系统提示词中明确工具的使用场景：

```
### 可使用工具及能力边界
1. rag_summarize：
   使用场景：当回答用户问题需要补充专业信息时调用
2. get_weather：
   使用场景：用户询问天气或环境适配问题时调用
3. fetch_external_data：
   使用场景：用户需要生成个人报告时调用
```

---

## 6. 工具错误处理

### 6.1 基础错误处理

```python
@tool
def fetch_external_data(user_id: str, month: str) -> str:
    """获取用户使用记录"""
    generate_external_data()
    
    try:
        return str(external_data[user_id][month])
    except KeyError:
        logger.warning(f"未找到用户 {user_id} 在 {month} 的记录")
        return ""
    except Exception as e:
        logger.error(f"获取数据失败: {str(e)}", exc_info=True)
        return "获取数据失败，请稍后重试"
```

### 6.2 工具返回值规范

建议：
- 始终返回字符串
- 错误信息要清晰
- 成功时返回结构化数据（JSON 字符串）

```python
# 好的返回值
return json.dumps({"status": "success", "data": {...}})

# 不好的返回值
return {"status": "success"}  # 不是字符串
```

---

## 7. 总结

本节课学习了：

1. **@tool 装饰器**: 定义 Agent 工具的核心方式
2. **工具参数**: 支持有参数、无参数、多参数
3. **类型注解**: 自动生成工具 schema
4. **工具设计模式**: 注册、选择、优先级
5. **错误处理**: 工具的健壮性设计

---

## 代码练习

### 练习 5.1：计算器工具

创建一个计算器工具：

```python
@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果
    
    Args:
        expression: 数学表达式，如 "2+3*5"
        
    Returns:
        计算结果
    """
    # 安全计算：只能使用加减乘除
    # 提示：使用 eval() 并限制运算符
    pass
```

### 练习 5.2：搜索工具

创建一个网页搜索工具：

```python
@tool
def web_search(query: str, num_results: int = 5) -> str:
    """搜索网页内容
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量，默认5
        
    Returns:
        搜索结果列表
    """
    # 提示：可以调用搜索引擎 API
    pass
```

### 练习 5.3：数据库查询工具

创建一个模拟的数据库查询工具：

```python
@tool
def query_user_orders(user_id: str, status: str = None) -> str:
    """查询用户订单
    
    Args:
        user_id: 用户ID
        status: 订单状态，可选值 pending/shipped/delivered
        
    Returns:
        订单列表
    """
    # 模拟数据库
    mock_db = {...}
    # 实现查询逻辑
    pass
```

### 练习 5.4：组合工具

创建一个工具，组合多个工具的能力：

```python
@tool
def smart_search(query: str) -> str:
    """智能搜索，同时搜索知识库和网页
    
    Args:
        query: 搜索关键词
        
    Returns:
        综合搜索结果
    """
    # 1. 先尝试 RAG 搜索
    # 2. 如果 RAG 结果不足，补充网页搜索
    # 3. 合并结果返回
    pass
```

---

## 下节课预告

下一课我们将学习 **中间件（Middleware）机制**，包括：
- @wrap_tool_call 工具拦截
- @before_model 模型调用前拦截
- @dynamic_prompt 动态提示词切换

---

## 相关资源

- [LangChain Tools 文档](https://python.langchain.com/docs/modules/agents/tools/)
- [Tool Calling 指南](https://python.langchain.com/docs/modules/model_io/chat/tools/)
- [Pydantic Tools](https://python.langchain.com/docs/modules/model_io/chat/tools/#using-pydantic-classes)
