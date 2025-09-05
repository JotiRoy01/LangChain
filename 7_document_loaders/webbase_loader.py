from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
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


prompt = PromptTemplate(
    template="Answer the following questerion \n{question} from the following text- \n{text}",
    input_variables=["question","text"]
)

parser = StrOutputParser()



url = "https://www.amazon.com/Apple-iPhone-Version-128GB-Titanium/dp/B0DNT1GPC1/ref=sr_1_1?dib=eyJ2IjoiMSJ9.p-dSX6UYKzMSo9opaFCrqXfmScKUr14Uqad1KWGUvM6l2wqt5u6e9-zJkTxhr198yOYcm226mO3wKcnCueXZOOVXt3wWULaZB05WSgFqYCPjDK8n_KA3zg-EprTKqm5ZUyIPyLEMoDLe3n0_3vWJEFh5A-vC6df7h7I78e9W_wXZk_B4GaCqLSFPjUqnpep7l4q8dkXIETsS3LA4YFZ5RO5_oMMA_imJuqN8FiPye-A.EPp3Sq6fc5Pc4ZNS9gWwHYw-3a7zE6XkK0tC5k5pmwo&dib_tag=se&keywords=iphone+16&qid=1756977579&sr=8-1"
loder = WebBaseLoader(url)

docs = loder.load()
print(len(docs))
print(docs[0].page_content)

chain = prompt | model | parser

result = chain.invoke({"question":"what is the product that we are talking about?","text":docs[0].page_content})

print(result)