from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

import os 

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "sequential LLM app"

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"],
)

model1 = ChatMistralAI(model="mistral-small-latest",temperature=0.3)
model2 = ChatMistralAI(model="mistral-small-latest",temperature=0.7)
config = {
    'run_name': "ye_sequential_chain_hai_ji",
    "tags": ["llm app","report generation"],
    "metadata": {
        "model1": "gpt-1",
        "model2": 'gpt-10',
        "parser": "string wala output parser"
    }
}

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

result = chain.invoke({"topic": "Unemployment in India"},config = config)

print(result)
