from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel) :
    name: str = "joti"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10)

#new_student = {"name":"JOTI"}
new_student = {"name":"JOTI", "email":"abce@gamil.com", "cgpa": 5}
#new_student = {}

student = Student(**new_student)
student_dict = dict(student)
print(student_dict["name"])

student_json = student.model_dump_json()
print(student)