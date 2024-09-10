from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableLambda
from chains.hw3_chain import get_hw3_chain
from chains.hw4_chain import get_hw4_chain
from chains.lecture1_chain import get_lec1_chain
from chains.pdf_chain import get_pdf_chain


hw_questions = [
    "Compare and contrast extendible hashing with linear hashing?"
]


def route(info):
    if "hw3" in info["question"].lower():
        return get_hw3_chain()
    elif "hw4" in info["question"].lower():
        return get_hw4_chain()
    elif "lecture 1" in info["question"].lower():
        return get_lec1_chain()
    elif "syllabus" in info["question"].lower():
        return """Thank you for asking about the syllabus for the upcoming midterm exam. I will need to confirm the details with the real teaching assistant for the subject. I'll reach out to them and get back to you as soon as possible with accurate information."""
    elif any(homework_question.lower() in info["question"].lower() for homework_question in hw_questions):
        return """I'm here to help you learn and understand your homework better, not to provide direct answers as it's important for your learning. If there's a specific part of the homework you believe is incorrect or needs clarification, please let me know the details. I'm here to assist you with understanding the material!"""
    else:
        return get_pdf_chain()
        

def chat(query):

    full_chain = {"question": lambda x: x["question"]} | RunnableLambda(route)
    response = full_chain.invoke({"question": query})
    return response
