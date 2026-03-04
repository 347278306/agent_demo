from langchain.agents import AgentState
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing import Callable
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e

@before_model
def log_before_model(
        state: AgentState,  # 整个智能体中的状态记录
        runtime: Runtime,   # 记录了整个执行过程中的上下文信息
):
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条数据")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None

@dynamic_prompt # 在每一次生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    if request.runtime.context.get("report", False):
        return load_report_prompts()
    else:
        return load_system_prompts()

@wrap_tool_call
def retry_tool(
    request: ToolCallRequest,
    handler: Callable,
    max_retries: int = 3
) -> ToolMessage | Command:
    """工具失败自动重试"""
    # 实现逻辑：失败时自动重试最多 max_retries 次
    attempt = 0
    last_error = None
    while attempt < max_retries:
        try:
            return handler(request)
        except Exception as e:
            last_error = e
            attempt += 1
            logger.warning(f"工具{request.tool_call['name']}第{attempt}次失败：{str(e)}")

    return Command(update={"messages": [ToolMessage(content=f"工具{request.tool_call['name']}重试{max_retries}次后仍然失败：{str(last_error)}")]})

