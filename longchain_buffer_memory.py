from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_debug # For debugging
from operator import itemgetter

from utils import LLMUtils

set_debug(True)

llm = LLMUtils().get_openai_llm()

messages = [
        "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
        "Qual é o melhor período do ano para visitar em termos de clima?",
        "Quais tipos de atividades ao ar livre estão disponíveis?",
        "Alguma sugestão de acomodação eco-friendly por lá?",
        "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
        "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

long_chat = ""

memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, verbose=True, memory=memory)

for message in messages:
    response = conversation.predict(input=message)
    print(response)