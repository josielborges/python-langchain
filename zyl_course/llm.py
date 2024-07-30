import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini',
                 api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=1,
                 verbose=True)

# SIMPLE
# response = llm.invoke("Hello, how are you")
# print(response)

# BATCH
# response = llm.batch(["Hello, how are you", "How much is 2 + 2?"])
# print(response)

# STREAM
response = llm.stream("Write a poem about python")  # return in chunks
for chunk in response:
    print(chunk.content, end='', flush=True)
