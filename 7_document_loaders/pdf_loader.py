from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["HF_HOME"] = 'D:/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache/transformers"

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

loader = PyPDFLoader("rag_application.pdf")
docs = loader.load()
print(len(docs))