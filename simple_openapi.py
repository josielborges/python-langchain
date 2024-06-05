import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

days = 7
children_size = 2
activity = 'praia'

prompt = f"Crie um roteiro de viagem de {days} dias, para uma família com {children_size} crianças, que gostam de {activity}."

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "system",
            "content": prompt
        }
    ],
)

print(response.choices[0].message.content)
