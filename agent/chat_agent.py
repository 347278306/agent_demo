from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

from model.factory import chat_model


@tool(description="计算数学表达式结果")
def calculator(expression: str) -> str:
    """计算数据表达式"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

class ChatAgent():
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt="你是一个数学助手，可以使用计算器工具。",
            tools=[calculator]
        )
        self.history: list[dict] = []   # 存储对话记录

    def chat(self, user_input: str):
        # 将历史消息添加到输入中
        messages = self._build_messages(user_input)
        response = self.agent.invoke({"messages": messages})
        assistant_message = response["messages"][-1]

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": assistant_message.content})

        return assistant_message.content

    def _build_messages(self, user_input: str) -> list:
        messages = []
        for msg in self.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))
        return messages

    def clear_history(self):
        self.history = []

if __name__ == '__main__':
    agent = ChatAgent()
    print(agent.chat("2+3等于多少？"))
    print(agent.chat("刚才的结果乘以10是多少？"))
