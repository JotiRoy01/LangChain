from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ("system","You are a helpful {domain} expert"),
    ("human","Explain in simple terms, what is {topic}")
    # SystemMessage(content="You are a helpful {domain} expert"),
    # HumanMessage(content="Explain in simple terms, what is {topic}")
])

prompt = chat_template.invoke({"domain":"criket", "topic":"Dusra"})
print(prompt)