# AGENTS.md

This file provides guidelines for agentic coding agents working in this repository.

## Project Overview

Python-based RAG (Retrieval Augmented Generation) agent system using LangChain, Chroma vector store, and DashScope (Alibaba) for embeddings and chat models.

## Project Structure

```
agent_demo/
├── agent/tools/        # Agent tools implementation
├── config/             # YAML configuration files (agent.yaml, chroma.yaml, prompts.yaml, rag.yaml)
├── data/               # Data files and external data
├── logs/               # Application logs
├── model/              # Model factory (ChatModel, Embeddings)
├── prompts/            # Prompt templates
├── rag/                # RAG service and vector store
└── utils/              # Utility modules (config_handler, file_handler, logger_handler, path_tool, prompt_loader)
```

## Build/Lint/Test Commands

### Running Python Modules

```bash
python -m module_name           # Run any module
python rag/rag_service.py       # Run RAG service
python rag/vector_store.py      # Load documents to vector store
python agent/tools/agent_tools.py  # Test agent tools
```

### Running Tests

```bash
pytest                          # Run all tests
pytest tests/test_file.py       # Run specific test file
pytest tests/test_file.py::test_function_name  # Run specific test
pytest -k "test_pattern"        # Run tests matching pattern
```

### Linting and Type Checking

```bash
ruff check .                    # Lint with ruff
ruff check --fix .              # Auto-fix linting issues
black .                         # Format code
mypy .                          # Type checking
```

### Virtual Environment

```bash
.venv\Scripts\activate          # Activate (Windows)
pip install -r requirements.txt # Install dependencies
```

## Code Style Guidelines

### Imports

Group imports in order: stdlib, third-party, local. Use absolute imports.

```python
import os
import hashlib

from langchain_core.documents import Document
from langchain_chroma import Chroma

from model.factory import chat_model
from utils.config_handler import rag_config
```

### Formatting

- 4 spaces for indentation
- Max line length: 120 characters
- snake_case: variables, functions
- PascalCase: classes
- UPPER_SNAKE_CASE: constants

### Type Hints

Always use type hints:

```python
def get_file_md5_hex(filepath: str) -> str | None: ...
def listdir_with_allowed_type(path: str, allowed_types: tuple[str]) -> list[str]: ...
class RagSummarizeService:
    def rag_summarize(self, query: str) -> str: ...
```

### Error Handling

- Use try/except for operations that may fail
- Log errors with context and `exc_info=True`
- Re-raise exceptions after logging when appropriate

```python
try:
    documents = get_file_document(path)
except Exception as e:
    logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
    continue
```

### Logging

Use `utils.logger_handler` module:

```python
from utils.logger_handler import logger

logger.info(f"[加载知识库]{path}内容加载成功")
logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}")
logger.error(f"计算文件{filepath}md5失败， {str(e)}", exc_info=True)
```

### Path Handling

Use `utils.path_tool.get_abs_path()` for path resolution:

```python
from utils.path_tool import get_abs_path

external_data_path = get_abs_path(agent_config["external_data_path"])
```

### Configuration

Store config in YAML files under `config/`. Access via `utils.config_handler`:

```python
from utils.config_handler import rag_config, chroma_config, agent_config

model_name = rag_config["chat_model_name"]
```

### Agent Tools

Use `@tool` decorator from LangChain:

```python
from langchain_core.tools import tool

@tool(description="工具功能描述")
def tool_name(param: str) -> str:
    """工具的具体说明"""
    return result
```

### Documentation

Add docstrings to all public functions/classes. Use Chinese for user-facing strings, English for code comments. Specify UTF-8 encoding when reading/writing files.

```python
def get_abs_path(relative_path: str) -> str:
    """
    Convert relative path to absolute path.
    :param relative_path: Relative path from project root
    :return: Absolute path
    """
```

## Development Workflow

1. Activate virtual environment before development
2. Run linting before committing
3. Add tests for new functionality
4. Use logging for debugging
5. Handle errors gracefully with proper error messages
