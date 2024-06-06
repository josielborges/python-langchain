import os

from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI


class LLMUtils:
    __llm = None
    __embeddings = None
    __temperature = None
    __api_key = os.getenv('OPENAI_API_KEY')

    def __init__(self, model='openai', temperature=1.0):
        load_dotenv()
        self.__temperature = temperature
        self.__llm = self.get_llm(model)
        self.__embeddings = self.get_embeddings(model)

    def get_openai_llm(self) -> ChatOpenAI:
        llm = ChatOpenAI(
            temperature=self.__temperature,
            model='gpt-3.5-turbo',
            api_key=self.__api_key
        )
        return llm

    def get_ollama_llm(self) -> Ollama:
        llm = Ollama(
            temperature=self.__temperature,
            model="phi3"
        )
        return llm

    def get_llm(self, model):
        if model == 'openai':
            return self.get_openai_llm()
        elif model == 'ollama':
            return self.get_ollama_llm()
        else:
            raise Exception('Invalid model')

    def get_embeddings(self, model):
        if model == 'openai':
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()
        elif model == 'ollama':
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings()
        else:
            raise Exception('Invalid model')

    def getLLM(self):
        return self.__llm

    def getEmbeddings(self):
        return self.__embeddings
