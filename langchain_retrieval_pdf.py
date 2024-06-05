from langchain.globals import set_debug  # For debugging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from utils import LLMUtils

# set_debug(True)

llm = LLMUtils().get_openai_llm()

loaders = [
    PyPDFLoader("data/GTB_black_Nov23.pdf"),
    PyPDFLoader("data/GTB_gold_Nov23.pdf"),
    PyPDFLoader("data/GTB_platinum_Nov23.pdf"),
    PyPDFLoader("data/GTB_standard_Nov23.pdf")
]

documents = []

for loader in loaders:
    documents.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
texts = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever = db.as_retriever())

query = 'Como devo proceder caso tenha um item comprado roubado'
result = qa_chain({'query': query})
print(result)
