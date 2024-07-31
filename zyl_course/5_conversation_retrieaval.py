import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.globals import set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

set_debug(True)
load_dotenv()


def get_documents_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    split_docs = splitter.split_documents(documents)
    print(len(split_docs))
    return split_docs


def create_db(docs):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embedding)
    return vector_store


def create_chain(vector_store):
    llm = ChatOpenAI(model='gpt-4o-mini',
                     api_key=os.getenv("OPENAI_API_KEY"),
                     temperature=0.4,
                     verbose=True)

    # TO SIMPLE REQUESTS
    # prompt = ChatPromptTemplate.from_template('''
    # Responda a pergunta to usuário.
    # Context: {context},
    # Pergunta: {input} ''')

    prompt = ChatPromptTemplate.from_messages([
        ('system', "Responda o usuario baseando-se no contexto: {context}"),
        (MessagesPlaceholder(variable_name='chat_history')),
        ('human', "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # This simply get the input and pass to retriever to get the most important parts
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # it will get the 2 most important parts of document
    # retrieval_chain = create_retrieval_chain(
    #     retriever,  # these parts will automatically be inputted at prompt in context parameter
    #     chain
    # )

    # It will generate a new input based on user input and history to finetune the question
    # pay attention on token usage
    retriever_prompt = ChatPromptTemplate.from_messages([
        (MessagesPlaceholder(variable_name='chat_history')),
        ('human', "{input}"),
        ('human',
         "Dada a conversa acima, gere uma consulta de pesquisa para obter informações relevantes para a conversa.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        # retriever,  # this parts will automatically be inputted at prompt in context parameter
        history_aware_retriever,
        # this will input a new input based on user input and history, to the retriever get the best parts
        chain
    )

    return retrieval_chain


def process_chat(chain, question, chat_history):
    response = chain.invoke(
        {
            "input": question,  # need to be "input"
            'chat_history': chat_history
            # "context": docs # not necessary anymore, because what is as explained above
        })

    return response['answer']


if __name__ == '__main__':
    docs = get_documents_from_url("https://python.langchain.com/v0.1/docs/expression_language/")
    vector_store = create_db(docs)
    chain = create_chain(vector_store)

    chat_history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            break

        response = process_chat(chain, user_input, chat_history)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print('Assistant: ', response)
