from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from chains.utils import rag_chain_from_docs

def get_lec1_chain():
    embedding_fn = OpenAIEmbeddings()
    lec1_db = FAISS.load_local("./data/video_index/Lec_1_db", embedding_fn, allow_dangerous_deserialization=True)
    lec1_retriever = lec1_db.as_retriever(search_kwargs={"k":2})

    lec1_prompt = PromptTemplate.from_template(
            """You are an expert in answering questions related to Lecture 1 of Database Systems. \
    Respond to the following question using only the provided context:
    If the context do not contain an answer but lead to a resource mention the resource as the final answer.
    Do not use your internal memory to answer the question. Please elaborate on the answer. 

    Context: {context}

    Question: {question}

    Answer:"""
    )

    output_parser = StrOutputParser()
    llm = ChatOpenAI()

    lec1_chain = RunnableParallel(
        context=(lambda x: x["question"]) | lec1_retriever,
        question=(lambda x: x["question"])
    ).assign(answer=rag_chain_from_docs(lec1_prompt, llm, output_parser))
    
    return lec1_chain