from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

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

#Schema
class Review(BaseModel) :
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Great! I was a bit weary as I had previously ordered a “unlocked” apple phone from a different seller and turned out it wasn't locked. 
             This one was and I was able to transfer an existing number to this phone through Xfinity Mobile. I didn't need to call customer service, process was simple and straightforward. 
             The phone was in good condition with no dings or scratches, shipping was quick, and I paid half the cost vs. had I purchased new.""")


print(result)