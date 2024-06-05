import os

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from utils import LLMUtils

__update_embeddings = False

# set_debug(True)

llm = LLMUtils().get_openai_llm()
embeddings = OpenAIEmbeddings()


def update_embeddings():
    loaders = [
        PyPDFLoader("data/GTB_black_Nov23.pdf"),
        PyPDFLoader("data/GTB_gold_Nov23.pdf"),
        PyPDFLoader("data/GTB_platinum_Nov23.pdf"),
        PyPDFLoader("data/GTB_standard_Nov23.pdf")
    ]

    documents = []

    for loader in loaders:
        documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    db = FAISS.from_documents(texts, embeddings)
    db.save_local("db/faiss-index")


def load_embeddings():
    return FAISS.load_local("db/faiss-index", embeddings, allow_dangerous_deserialization=True)


if __update_embeddings or not os.path.exists("db/faiss-index"):
    update_embeddings()

db = load_embeddings()

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

query = 'Como devo proceder caso tenha um item comprado roubado'
result = qa_chain({'query': query})
print(result)
