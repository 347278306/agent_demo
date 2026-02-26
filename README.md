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
├── agent/              # Agent 工具实现
│   └── tools/
│       └── agent_tools.py
├── config/             # YAML 配置文件
│   ├── agent.yaml
│   ├── chroma.yaml
│   ├── prompts.yaml
│   └── rag.yaml
├── data/               # 数据文件
├── logs/               # 日志文件
├── model/              # 模型工厂
│   └── factory.py
├── prompts/            # Prompt 模板
├── rag/                # RAG 服务和向量存储
│   ├── rag_service.py
│   └── vector_store.py
└── utils/              # 工具模块
    ├── config_handler.py
    ├── file_handler.py
    ├── logger_handler.py
    ├── path_tool.py
    └── prompt_loader.py
```

## 功能特性

- **RAG 问答**: 基于向量检索的智能问答
- **知识库管理**: 支持 TXT/PDF 文档加载和向量存储
- **Agent 工具**: 提供天气查询、用户信息获取等工具
- **外部数据集成**: 支持从外部 CSV 文件获取用户数据

## 快速开始

### 1. 安装依赖

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
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

### 5. 使用 Agent 工具

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
# 安装 pytest
pip install pytest

# 运行所有测试
pytest

# 运行指定测试
pytest tests/test_file.py
```

## 许可证

MIT License
