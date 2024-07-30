import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# set_debug(True)
load_dotenv()


# OLD TEST
# docA = Document(
#     page_content="LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. "
#                  "LCEL was designed from day 1 to support putting prototypes in production, with no code changes, "
#                  "from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully "
#                  "run LCEL chains with 100s of steps in production). To highlight a few of the reasons you might want
#                  to use LCEL.",
# )

def get_documents_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
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

    prompt = ChatPromptTemplate.from_template('''
    Responda a pergunta to usuário. 
    Context: {context},
    Pergunta: {input} ''')

    # chain = prompt | llm # for elaborated chains it is not too good
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # it will get the 2 most important parts of document
    retrieval_chain = create_retrieval_chain(
        retriever,  # these parts will automatically be inputted at prompt in context parameter
        chain
    )

    return retrieval_chain


docs = get_documents_from_url("https://python.langchain.com/v0.1/docs/expression_language/")
vector_store = create_db(docs)
chain = create_chain(vector_store)

response = chain.invoke(
    {
        "input": "What is LCEL?"  # need to be "input"
        # "context": docs # not necessary anymore, because what is as explained above
    })

print(response)
print(len(response['context']))  # number of docs/chunks
print(response['answer'])

# SIMPLE FROM DOCUMENTS
# docs = get_documents_from_url("https://python.langchain.com/v0.1/docs/expression_language/")
# response = chain.invoke(
#     {
#         "question": "What is LCEL?",
#         "context": docs
#     })
# print(response)

# References
# https://python.langchain.com/v0.1/docs/expression_language/
