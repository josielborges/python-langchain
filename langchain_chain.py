from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.globals import set_debug # For debugging
from utils import LLMUtils

set_debug(True)

llm = LLMUtils().get_openai_llm()

city_template = ChatPromptTemplate.from_template("Sugira uma cidade dado meu interesse por {interest}. A sua saída dee ser SOMENTE o nome da cidade. Cidade: ")
restaurant_template = ChatPromptTemplate.from_template("Sugira restaurantes populares entre locais em {city}")
cultural_template = ChatPromptTemplate.from_template("Sugira atividades e locais culturais em {city}")

city_chain = LLMChain(prompt=city_template, llm=llm)
restaurant_chain = LLMChain(prompt=restaurant_template, llm=llm)
cultural_chain = LLMChain(prompt=cultural_template, llm=llm)

chain = SimpleSequentialChain(chains=[city_chain, restaurant_chain, cultural_chain], verbose = True)

response = chain.invoke("praias")
print(response)