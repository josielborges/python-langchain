import os

from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

set_debug(True)

load_dotenv()

# Instantiate model
llm = ChatOpenAI(model='gpt-4o-mini',
                 api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=1,
                 verbose=True)

# SIMPLE WAY
# response = llm.invoke("Me conte uma piada sobre galinhas")
# print(response)

# WITH PROMPT TEMPLATE
# prompt = ChatPromptTemplate.from_template("Me conte uma piada sobre {subject}")
# chain = prompt | llm
# response = chain.invoke({"subject": "cachorros"})
# print(response)

# WITH MESSAGES
prompt = ChatPromptTemplate.from_messages([
    ('system', "Você é um contador de piadas e deve contar uma piada sobre o conteúdo a seguir."),
    ('human', "{input}"),
])
chain = prompt | llm
response = chain.invoke({"input": "cachorros"})
print(response)
