from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

days = 7
children_size = 2
activity = 'praia'

prompt_template = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {days} dias, para uma família com {children_size} crianças, que gostam de {activity}."
)

prompt = prompt_template.format(days=days,
                                children_size=children_size,
                                activity=activity)

llm = ChatOpenAI(
    temperature=0.7,
    model='gpt-3.5-turbo',
    api_key=os.getenv("OPENAI_API_KEY")
)

response = llm.invoke(prompt)
print(response.content)
