# Agent Demo

基于 LangChain + Chroma 向量库 + DashScope（阿里云）的 RAG 智能问答系统。

## 项目简介

本项目是一个 RAG（Retrieval Augmented Generation）智能问答 Agent 系统，使用 LangChain 框架构建，结合 Chroma 向量存储和阿里云 DashScope 大语言模型实现知识库问答功能。

## 技术栈

- **Python**: 3.10+
- **LangChain**: 大语言模型应用框架
- **Chroma**: 向量数据库
- **DashScope**: 阿里云通义千问模型服务
- **YAML**: 配置文件

## 项目结构

```
agent_demo/
├── app.py                      # 主应用程序入口
├── agent/                      # Agent 实现
│   ├── react_agent.py          # ReAct Agent 实现
│   └── tools/
│       ├── agent_tools.py      # Agent 工具定义
│       └── middleware.py       # 工具中间件
├── config/                     # YAML 配置文件
│   ├── agent.yaml
│   ├── chroma.yaml
│   ├── prompts.yaml
│   └── rag.yaml
├── data/                       # 数据文件
│   ├── external/               # 外部数据
│   │   └── records.csv
│   ├── 扫地机器人100问.pdf
│   ├── 扫地机器人选购指南.txt
│   ├── 扫地机器人维护保养.txt
│   ├── 扫拖一体机器人100问.txt
│   └── 扫地机器人故障排除.txt
├── logs/                       # 日志文件
├── model/                      # 模型工厂
│   └── factory.py
├── prompts/                    # Prompt 模板
│   ├── main_prompt.txt
│   ├── rag_summarize.txt
│   └── report_prompt.txt
├── rag/                        # RAG 服务和向量存储
│   ├── rag_service.py
│   └── vector_store.py
└── utils/                      # 工具模块
    ├── config_handler.py
    ├── file_handler.py
    ├── logger_handler.py
    ├── path_tool.py
    └── prompt_loader.py
```

## 功能特性

- **RAG 问答**: 基于向量检索的智能问答
- **ReAct Agent**: 支持推理和行动交替的 Agent 实现
- **知识库管理**: 支持 TXT/PDF 文档加载和向量存储
- **Agent 工具**: 提供天气查询、用户信息获取等工具
- **外部数据集成**: 支持从外部 CSV 文件获取用户数据
- **Web 应用**: 提供交互式 Web 界面

## 快速开始

> 注意：建议按顺序执行，先加载知识库后再运行 Web 应用或 RAG 问答。

### 1. 安装依赖

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装测试工具（可选）
pip install pytest
```

### 2. 配置环境

在 `config/` 目录下配置相关参数：

- `rag.yaml`: 模型名称配置
- `chroma.yaml`: 向量库配置
- `agent.yaml`: Agent 配置
- `prompts.yaml`: Prompt 模板路径

### 3. 加载知识库

```bash
python rag/vector_store.py
```

### 4. 运行 RAG 问答

```bash
python rag/rag_service.py
```

### 5. 运行 Web 应用

```bash
streamlit run app.py
```

### 6. 使用 Agent 工具

```bash
python agent/tools/agent_tools.py
```

## 配置说明

### rag.yaml

```yaml
chat_model_name: "qwen3-max"
embedding_model_name: "text-embedding-v4"
```

### chroma.yaml

包含向量库集合名称、持久化路径、文本分块大小等配置。

### agent.yaml

```yaml
external_data_path: data/external/records.csv
```

## 开发指南

### 代码规范

- 使用 4 空格缩进
- 使用类型提示
- 使用 snake_case 命名变量和函数
- 使用 PascalCase 命名类
- 使用中文编写用户面向的日志和错误信息

### 日志

日志保存在 `logs/` 目录下，使用 `utils.logger_handler` 模块获取日志实例：

```python
from utils.logger_handler import logger

logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")
```

### 路径处理

使用 `utils.path_tool.get_abs_path()` 获取绝对路径：

```python
from utils.path_tool import get_abs_path

abs_path = get_abs_path("data/test.txt")
```

## 运行测试

```bash
# 运行所有测试
pytest

# 运行指定测试文件
pytest tests/test_file.py

# 运行指定测试函数
pytest tests/test_file.py::test_function_name

# 运行匹配模式的测试
pytest -k "test_pattern"
```

### 代码检查

```bash
# 代码格式检查
ruff check .

# 自动修复代码问题
ruff check --fix .

# 代码格式化
black .

# 类型检查
mypy .
```

## 许可证

MIT License
