from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

os.environ["HF_HOME"] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)
model = ChatHuggingFace(llm = llm)

# 1st promt -> detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)
# 2nd prompt -> summary
template2 = PromptTemplate(
    template="write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic":"black hole"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text":result1.content})
result2 = model.invoke(prompt2)

print(result1.content)