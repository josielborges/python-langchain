from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from utils import LLMUtils

# set_debug(True)

llm = LLMUtils().get_openai_llm()

text_loader = TextLoader("data/GTB_gold_Nov23.txt", encoding='utf-8')
documents = text_loader.load()
splitter = CharacterTextSplitter(separator='.', chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

query = 'Como devo proceder caso tenha um item comprado roubado'
result = qa_chain({'query': query})
print(result)
