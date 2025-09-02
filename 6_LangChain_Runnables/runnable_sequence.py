from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

os.environ["HF_HOME"] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)
#initialize the LLM
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template="Write a joke about{topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="explain the following joke - {text}",
    input_variables=["text"]
)


parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
result = chain.invoke({"topic":"AI"})
print(result)