import os

from dotenv import load_dotenv
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.globals import set_debug
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI

set_debug(True)

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini',
                 api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=1,
                 verbose=True)

prompt = ChatPromptTemplate.from_messages([
    ('system', "Você é um adorável assistente chamado Max"),
    ('human', "{input}"),
    MessagesPlaceholder(variable_name='agent_scratchpad'),  # it will include the agent_scratchpad in the prompt
])

search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
tools = [search]

agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools  # these tools will be called if the llm can't respond a user question
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools
)

response = agent_executor.invoke({
    "input": "Qual o próximo jogo do Vasco da Gama?"
})

print(response['output'])

# references
# https://python.langchain.com/docs/modules/agents/agent_types
