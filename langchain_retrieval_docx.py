import os

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_text_splitters import CharacterTextSplitter

from utils import LLMUtils

# set_debug(True)

is_update_embeddings = False
faiss_file_path = "db/nps/faiss-index"

llm_factory = LLMUtils('openai')
llm = llm_factory.getLLM();
embeddings = llm_factory.getEmbeddings();


def update_embeddings():
    loaders = []

    for np_file in os.listdir('nps'):
        if np_file.startswith('.~'):
            continue
        if np_file.endswith('.docx'):
            loaders.append(Docx2txtLoader("nps/" + np_file))
        if np_file.endswith('.pdf'):
            loaders.append(PyPDFLoader("nps/" + np_file))

    documents = []

    for loader in loaders:
        documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(faiss_file_path)


def load_embeddings():
    return FAISS.load_local(faiss_file_path, embeddings, allow_dangerous_deserialization=True)


class OutputNps(BaseModel):
    np = Field("np no formato NP-XXX")
    text = Field("texto de resposta")


output_parser = JsonOutputParser(pydantic_object=OutputNps)

if is_update_embeddings or not os.path.exists(faiss_file_path):
    print('Updating embeddings')
    update_embeddings()

db = load_embeddings()

prompt = PromptTemplate(template='''
                        Você é um assistente de inteligência artificial especializado em responder perguntas com base em documentos fornecidos. 
                        Por favor, seja preciso e claro nas respostas.
                        Inclua a NP na resposta no formato NP-XXX-XXXX.
                        Pergunta: {query}.
                        {output_format}''',
                        input_variables=["query"],
                        partial_variables={"output_format": output_parser.get_format_instructions()})

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

while True:
    query = input('O que você gostaria de saber? [Em branco para finalizar] : ')

    if query == '':
        break

    result = qa_chain({'query': prompt.format(query=query)})
    print(result['result'])
