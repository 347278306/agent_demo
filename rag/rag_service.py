from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt

class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        self.conversation_history: list[dict] = []

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )

    def _build_context_with_history(self, query: str) -> str:
        """构件包含对话历史的上下文"""
        history_text = ""

        # 添加历史对话
        for msg in self.conversation_history[-5:]:  # 保留最近5轮
            role = msg["role"]
            context = msg["context"]
            history_text += f"{role}: {context}\n"

        # 添加当前问题
        history_text += f"用户：{query}\n"
        return history_text

    def chat(self, query: str) -> str:
        # 1. 获取检索结果
        context_decs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in context_decs])

        # 2. 构件完整的上下文（包含历史）
        full_context = self._build_context_with_history(query)

        # 3. 调用链
        result = self.chain.invoke({"input": query, "context": context, "history": full_context})

        # 4. 保存对话历史
        self.conversation_history.append({"role": "用户", "context": query})
        self.conversation_history.append({"role": "助手", "context": result})

        return result


class Answer(BaseModel):
    answer: str
    source: list[str]

class RagJsonService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.parser = PydanticOutputParser(pydantic_object=Answer)

        # 在提示此种嵌入格式说明
        self.prompt_template = PromptTemplate.from_template(
            """基于一下参考资料回答问题。
            参考资料：{context}
            问题：{input}
            {format_instructions}""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt_template | chat_model | self.parser

    def rag_summarize(self, query: str) -> Answer:
        docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return self.chain.invoke({"input": query, "context": context})

if __name__ == '__main__':
    # rag = RagSummarizeService()
    # rag = RagJsonService()
    # res = rag.rag_summarize("扫地机器人的电池一般能用多久？")
    # print(res)

    # 使用示例
    rag = RagSummarizeService()
    # 第一轮
    response1 = rag.chat("扫地机器人电池能用多久？")
    print(response1)
    # 第二轮（带上下文）
    response2 = rag.chat("那充满电需要多长时间？")
    print(response2)
