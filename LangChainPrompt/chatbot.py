from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


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

chat_history = [
    SystemMessage(content="Your are a helpful AI assistant")
]

while True :
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input =="exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)