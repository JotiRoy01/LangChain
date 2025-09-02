from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

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

def word_counter(text) :
    return len(text.split())

# runable_word_counter = RunnableLambda(word_counter)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a joke about{topic}",
    input_variables=["topic"]
)
joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(word_counter)
})

# parallel_chain = RunnableParallel({
#     "joke":RunnablePassthrough(),
#     "word_count":RunnableLambda(lambda x : len(x.split()))
# })

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

final_chain.invoke({"topic":"AI"})
