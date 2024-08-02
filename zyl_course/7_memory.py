import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# from langchain.globals import set_debug
# set_debug(True)

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini',
                 api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=1,
                 verbose=True)

prompt = ChatPromptTemplate.from_messages([
    ('system', "Você é um adorável assistente chamado Max"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', "{input}")
])

history = UpstashRedisChatMessageHistory(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
    session_id="chat1",
    ttl=0  # time to live in seconds
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

if __name__ == '__main__':

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            break

        msg = {
            "input": user_input
        }

        response = chain.invoke(msg)

        print('Assistant: ', response['text'])
