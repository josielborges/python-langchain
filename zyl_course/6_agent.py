import os

from dotenv import load_dotenv
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.globals import set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

set_debug(True)

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini',
                 api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=1,
                 verbose=True)


def create_retriever_tool_internal():
    loader = WebBaseLoader("https://python.langchain.com/v0.1/docs/expression_language/")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    split_docs = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embedding)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # it will get the 3 most important parts of document

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="retriever",
        description="Use essa tool para pesquisar informações sobre Langchain Expression Language (LCEL)"
    )

    return retriever_tool


prompt = ChatPromptTemplate.from_messages([
    ('system', "Você é um adorável assistente chamado Max"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', "{input}"),
    MessagesPlaceholder(variable_name='agent_scratchpad'),  # it will include the agent_scratchpad in the prompt
])

search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
retriever_tool = create_retriever_tool_internal()

tools = [search, retriever_tool]

agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools  # these tools will be called if the llm can't respond a user question
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools
)


def process_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response['output']


if __name__ == '__main__':
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agent_executor, user_input, chat_history)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print('Assistant: ', response)

# references
# https://python.langchain.com/docs/modules/agents/agent_types
