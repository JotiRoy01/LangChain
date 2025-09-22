from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = """The determinant of a square matrix is a scalar value, a single number, that encodes both algebraic and geometric information about the matrix and the linear transformation it represents. Geometrically, it represents the scaling factor for area or volume under that transformation, while algebraically, it is crucial for solving systems of linear equations and finding the inverse of a matrix. 
# Geometric Meaning
# Scaling Factor:
# The determinant tells you how much a linear transformation stretches or shrinks space. For example, a determinant of 3 means the area or volume scales by a factor of three. 
# Orientation:
# A positive determinant means the transformation preserves the orientation of space (no flipping), while a negative determinant indicates a flip or inversion of space. 
# Zero Determinant:
# If a matrix has a determinant of zero, it means the transformation collapses space into a smaller dimension (e.g., a 3D object into a 2D plane or a line), causing the volume to become zero"""


# splitter = CharacterTextSplitter(
#     chunk_size = 100,
#     chunk_overlap = 0,
#     separator=""
# )

# result = splitter.split_text(text=text)
# print(result)

loader = PyPDFLoader("rag_application.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=""
)

result = splitter.split_documents(documents=docs)
print(result)