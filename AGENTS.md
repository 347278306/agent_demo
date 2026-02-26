# AGENTS.md

This file provides guidelines for agentic coding agents working in this repository.

## Project Overview

This is a Python-based RAG (Retrieval Augmented Generation) agent system using LangChain, Chroma vector store, and DashScope (Alibaba) for embeddings and chat models.

## Project Structure

```
agent_demo/
├── agent/              # Agent tools implementation
│   └── tools/
│       └── agent_tools.py
├── config/             # YAML configuration files
│   ├── agent.yaml
│   ├── chroma.yaml
│   ├── prompts.yaml
│   └── rag.yaml
├── data/               # Data files and external data
├── logs/               # Application logs
├── model/              # Model factory (ChatModel, Embeddings)
│   └── factory.py
├── prompts/            # Prompt templates
├── rag/                # RAG service and vector store
│   ├── rag_service.py
│   └── vector_store.py
└── utils/              # Utility modules
    ├── config_handler.py
    ├── file_handler.py
    ├── logger_handler.py
    ├── path_tool.py
    └── prompt_loader.py
```

## Build/Lint/Test Commands

### Running Python Files

```bash
# Run any Python module directly
python -m module_name

# Example: Run RAG service
python rag/rag_service.py

# Example: Run vector store
python rag/vector_store.py

# Example: Run agent tools
python agent/tools/agent_tools.py

# Example: Run utilities
python utils/logger_handler.py
python utils/config_handler.py
python utils/prompt_loader.py
python utils/path_tool.py
python utils/file_handler.py
python model/factory.py
```

### Running a Single Test

No tests exist in this project yet. To add tests:

```bash
# Install pytest
pip install pytest

# Run all tests
pytest

# Run a specific test file
pytest tests/test_file.py

# Run a specific test function
pytest tests/test_file.py::test_function_name

# Run tests matching a pattern
pytest -k "test_pattern"
```

### Linting and Type Checking

```bash
# Install linting tools
pip install ruff black mypy

# Run ruff (linting)
ruff check .

# Run ruff with auto-fix
ruff check --fix .

# Run black (formatting)
black .

# Run mypy (type checking)
mypy .
```

### Virtual Environment

The project uses `.venv` for dependency management:

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Code Style Guidelines

### Imports

- Use absolute imports within the project
- Group imports in the following order: stdlib, third-party, local
- Use `from` imports for readability

```python
# Standard library
import os
import hashlib

# Third-party
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Local project
from model.factory import chat_model
from utils.config_handler import rag_config
```

### Formatting

- Use 4 spaces for indentation (PEP 8)
- Maximum line length: 120 characters
- Use snake_case for variable and function names
- Use PascalCase for class names
- Use ALL_CAPS for constants

### Type Hints

Always use type hints for function parameters and return types:

```python
def get_file_md5_hex(filepath: str) -> str | None:
    """Get MD5 hex string of a file."""
    ...

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]) -> list[str]:
    """Return list of files with allowed types."""
    ...

class RagSummarizeService:
    def rag_summarize(self, query: str) -> str:
        ...
```

### Naming Conventions

- Variables: `snake_case` (e.g., `external_data`, `user_ids`)
- Functions: `snake_case` (e.g., `get_file_md5_hex`, `load_rag_prompts`)
- Classes: `PascalCase` (e.g., `RagSummarizeService`, `VectorStoreService`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `LOG_ROOT`, `DEFAULT_LOG_FORMAT`)
- File names: `snake_case.py` (e.g., `rag_service.py`, `config_handler.py`)

### Error Handling

- Use try/except blocks for operations that may fail
- Always log errors with appropriate context
- Include `exc_info=True` for detailed error traces when needed
- Re-raise exceptions after logging when appropriate

```python
try:
    documents = get_file_document(path)
except Exception as e:
    logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
    continue
```

### Logging

- Use the centralized logger from `utils.logger_handler`
- Include contextual information in log messages
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR

```python
from utils.logger_handler import logger

logger.info(f"[加载知识库]{path}内容加载成功")
logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
logger.error(f"计算文件{filepath}md5失败， {str(e)}", exc_info=True)
```

### Path Handling

- Always use `utils.path_tool.get_abs_path()` for path resolution
- Never hardcode absolute paths

```python
from utils.path_tool import get_abs_path

external_data_path = get_abs_path(agent_config["external_data_path"])
```

### Configuration

- Store all configuration in YAML files under `config/`
- Access config via `utils.config_handler` module
- Load config at module level for global access

```python
from utils.config_handler import rag_config, chroma_config, agent_config

# Usage
model_name = rag_config["chat_model_name"]
```

### Decorators

Use `@tool` decorator from LangChain for defining agent tools:

```python
from langchain_core.tools import tool

@tool(description="工具功能描述")
def tool_name(param: str) -> str:
    """工具的具体说明"""
    return result
```

### Documentation

- Add docstrings to all public functions and classes
- Use Chinese or English consistently (this codebase uses Chinese for user-facing docs)
- Include parameter and return type descriptions

```python
def get_abs_path(relative_path: str) -> str:
    """
    Convert relative path to absolute path.
    :param relative_path: Relative path from project root
    :return: Absolute path
    """
```

### Language

- Use Chinese for user-facing strings (error messages, logs, tool descriptions)
- Use English for code comments and variable names in most cases
- This codebase uses Chinese in some function docstrings - maintain consistency

### File Encoding

- Always specify UTF-8 encoding when reading/writing files

```python
with open(filepath, "r", encoding="utf-8") as f:
    ...
```

## Development Workflow

1. Activate virtual environment before development
2. Run linting before committing
3. Add tests for new functionality
4. Use logging for debugging
5. Handle errors gracefully with proper error messages

## Common Tasks

### Running the RAG Service

```bash
python rag/rag_service.py
```

### Loading Documents to Vector Store

```bash
python rag/vector_store.py
```

### Testing Agent Tools

```bash
python agent/tools/agent_tools.py
```
