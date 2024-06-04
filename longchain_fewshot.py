from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from utils import LLMUtils

llm = LLMUtils().get_openai_llm()

examples = [
    {
        "question": "Quem viveu mais, Muhammad Ali ou Alan Turing?",
        "answer": """
        São necessárias perguntas de acompanhamento: Sim.
        Pergunta: Quantos anos Muhammad Ali tinha quando morreu?
        Resposta intermediária: Muhammad Ali tinha 74 anos quando morreu.
        Pergunta: Quantos anos Alan Turing tinha quando morreu?
        Resposta intermediária: Alan Turing tinha 41 anos quando morreu.
        Então a resposta final é: Muhammad Ali
        """,
    },
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Pergunta: {question}\n{answer}"
)

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Pergunta: {input}",
    input_variables=["input"],
)

prompt = prompt_template.format(input="Quem foi o pai de Mary Ball Washington?")
resposta = llm.invoke(prompt)
print(resposta.content)
