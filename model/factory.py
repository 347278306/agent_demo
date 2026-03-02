from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from utils.config_handler import rag_config

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        model_name = rag_config["chat_model_name"]
        if not model_name:
            raise ValueError("chat_model_name未配置")
        return ChatTongyi(model=model_name)

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])

class OpenAIChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        return ChatOpenAI(model=rag_config["open_ai_name"])

chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()

def get_chat_model():
    """根据环境变量返回不同模型"""
    import os
    provider = os.environ.get("MODEL_PROVIDER", "dashscope")
    if provider == "openai":
        return OpenAIChatModelFactory().generator()
    else:
        return ChatModelFactory().generator()

if __name__ == '__main__':
    chat_model = get_chat_model()