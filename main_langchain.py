from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

days = 7
children_size = 2
activity = 'praia'

prompt = f"Crie um roteiro de viagem de {days} dias, para uma família com {children_size} crianças, que gostam de {activity}."

llm = ChatOpenAI(
    temperature = 0.7,
    model = 'gpt-3.5-turbo',
    api_key = os.getenv("OPENAI_API_KEY")
    )

response = llm.invoke(prompt)
print(response.content)