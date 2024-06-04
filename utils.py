from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os


class LLMUtils:
    __temperature = 0
    __model = ''
    __api_key = os.getenv('OPENAI_API_KEY')

    def __init__(self, temperature=1, model='gpt-3.5-turbo'):
        load_dotenv()
        self.__temperature = temperature
        self.__model = model

    def get_openai_llm(self) -> ChatOpenAI:
        llm = ChatOpenAI(
            temperature=self.__temperature,
            model=self.__model,
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
        if (model == 'openai'):
            return self.get_openai_llm()
        elif (model == 'ollama'):
            return self.get_ollama_llm()
        else:
            raise Exception('Invalid model')
