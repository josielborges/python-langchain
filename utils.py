from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

class LLMUtils() :

    __temperature = 0
    __model = ''
    __api_key = os.getenv('OPENAI_API_KEY')

    def __init__(self, temperature = 1, model = 'gpt-3.5-turbo'):
        load_dotenv()
        self.__temperature = temperature
        self.__model = model
        
    def get_openai_llm(self) -> ChatOpenAI:
        llm = ChatOpenAI(
            temperature = self.__temperature,
            model = self.__model,
            api_key = self.__api_key
        )
        return llm