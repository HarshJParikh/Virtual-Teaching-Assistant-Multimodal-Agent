from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from chains.utils import rag_chain_from_docs

def get_hw4_chain():
    embedding_fn = OpenAIEmbeddings()
    hw4_db = FAISS.load_local("./data/vector_index/hw4_db", embedding_fn, allow_dangerous_deserialization=True)
    hw4_retriever = hw4_db.as_retriever(search_kwargs={"k":2})

    hw4_prompt = PromptTemplate.from_template(
            """You are an expert in answering questions related to Homework 4 of Database Systems. \
    Respond to the following question using only the provided context:
    If the context do not contain an answer but lead to a resource mention the resource as the final answer.
    Do not use your internal memory to answer the question. Please elaborate on the answer. 

    Context: {context}

    Question: {question}

    Answer:"""
    )

    output_parser = StrOutputParser()
    llm = ChatOpenAI(api_key="")

    hw4_chain = RunnableParallel(
        context=(lambda x: x["question"]) | hw4_retriever,
        question=(lambda x: x["question"])
    ).assign(answer=rag_chain_from_docs(hw4_prompt, llm, output_parser))
    
    return hw4_chain