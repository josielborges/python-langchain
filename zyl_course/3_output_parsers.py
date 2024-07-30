import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# set_debug(True)
load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini',
                 api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=1,
                 verbose=True)


def call_no_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Você é um contador de piadas e deve contar uma piada sobre o conteúdo a seguir."),
        ('human', "{input}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"input": "cachorros"})
    return response


def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Você é um contador de piadas e deve contar uma piada sobre o conteúdo a seguir."),
        ('human', "{input}"),
    ])
    parser = StrOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({"input": "cachorros"})
    return response


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Retorne 5 sinônimos sobre a palagra a seguir. Retorne os resultado separato por vírgulas"),
        ('human', "{input}"),
    ])
    parser = CommaSeparatedListOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({"input": "cachorros"})
    return response


class Person(BaseModel):
    recipe: str = Field(description="o nome da receita")
    ingredients: list = Field(description="a lista de ingredientes da receita")


def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Retorne informações da seguinte a seguir.\nInstruções: {instructions}"),
        ('human', "{phrase}"),
    ])
    parser = JsonOutputParser(pydantic_object=Person)
    chain = prompt | llm | parser
    response = chain.invoke(
        {
            "phrase": "Quais os ingredientes de uma coxinha?",
            "instructions": parser.get_format_instructions()
         }
    )
    return response


print(type(call_no_output_parser()))
print(call_no_output_parser())
print(type(call_string_output_parser()))
print(call_string_output_parser())
print(type(call_list_output_parser()))
print(call_list_output_parser())
print(type(call_json_output_parser()))
print(call_json_output_parser())