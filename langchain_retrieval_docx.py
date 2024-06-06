from langchain.chains import RetrievalQA
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os

from utils import LLMUtils

# set_debug(True)

__update_embeddings = False
__faiss_file_path = "db/nps/faiss-index"

llm = LLMUtils().get_openai_llm()
embeddings = OpenAIEmbeddings()

def update_embeddings():
    loaders = []

    for np_file in os.listdir('nps'):
        if np_file.startswith('.~'):
            continue
        if np_file.endswith('.docx'):
            loaders.append(Docx2txtLoader("nps/"+np_file))
        if np_file.endswith('.pdf'):
            loaders.append(PyPDFLoader("nps/"+np_file))

    documents = []

    for loader in loaders:
        documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(__faiss_file_path)


def load_embeddings():
    return FAISS.load_local(__faiss_file_path, embeddings, allow_dangerous_deserialization=True)


if __update_embeddings or not os.path.exists(__faiss_file_path):
    update_embeddings()

db = load_embeddings()

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

while True:
    query = input('O que você gostaria de saber? [Em branco para finalizar] : ')

    if query == '':
        break

    result = qa_chain({'query': query})
    print(result['result'])
