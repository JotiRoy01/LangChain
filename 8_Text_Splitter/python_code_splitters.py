from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class ParentClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello from {self.name} in the parent class.")

class ChildClass(ParentClass):  # ChildClass inherits from ParentClass
    def __init__(self, name, age):
        super().__init__(name)  # Calls the __init__ of the parent class
        self.age = age

    def introduce(self):
        print(f"I am {self.name} and I am {self.age} years old.")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 0
)

#perform the split
chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])