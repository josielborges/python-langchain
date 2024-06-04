from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.globals import set_debug # For debugging

from utils import LLMUtils

set_debug(True)

class Destination(BaseModel):
    city = Field("cidade a visitar")
    reason = Field("motivo pelo qual é interessante visitar")

llm = LLMUtils().get_openai_llm()

output_parser = JsonOutputParser(pydantic_object=Destination)

city_template = PromptTemplate(template='''
                                        Sugira uma cidade dado meu interesse por {interest}.
                                        {output_format}
                                        ''',
                                        input_variables=["interest"],
                                        partial_variables={"output_format": output_parser.get_format_instructions()})

restaurant_template = ChatPromptTemplate.from_template("Sugira restaurantes populares entre locais em {city}")
cultural_template = ChatPromptTemplate.from_template("Sugira atividades e locais culturais em {city}")

city_chain = city_template | llm | output_parser
restaurant_chain = restaurant_template | llm | StrOutputParser()
cultural_chain = cultural_template | llm | StrOutputParser()

chain = city_chain | restaurant_chain | cultural_chain

response = chain.invoke({"interest" : "praias"})
print(response)