# Example: Importing a class or function from langchain_huggingface if available
# from langchain_huggingface import SomeClass

# If you meant to use HuggingFace integration with LangChain, you might want:
# from langchain_huggingface import HuggingFaceHub

# If you are not sure, you can comment out or remove the line until you install the package.
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

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



st.header("Reasearch Tool")

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All you Need", "BERT:Pre-training of Deep Bidirectional Transformers", "GPT-3:Language Model are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly","Technical","Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation langth",["Shot (1-2 paragraphs)", "Medium(3-5 paragraphs)", "Long(detailed explanation)"])

template = load_prompt('template.json')
#fill the placeholder
# prompt = template.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input
# })
if st.button("Summarize") :
    chain = template | model
    result = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    # prompt = template.invoke({
    # 'paper_input':paper_input,
    # 'style_input':style_input,
    # 'length_input':length_input
    # })
    st.write(result.content)