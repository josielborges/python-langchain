from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug  # For debugging
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel

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

# restaurant_template = ChatPromptTemplate.from_template("Sugira restaurantes populares entre locais em {city}")
# cultural_template = ChatPromptTemplate.from_template("Sugira atividades e locais culturais em {city}")

city_chain = LLMChain(prompt=city_template, llm=llm)
# restaurant_chain = LLMChain(prompt=restaurant_template, llm=llm)
# cultural_chain = LLMChain(prompt=cultural_template, llm=llm)

chain = SimpleSequentialChain(chains=[city_chain
                                    #   , restaurant_chain, cultural_chain
                                      ], verbose = True)

response = chain.invoke("praias")
print(response)