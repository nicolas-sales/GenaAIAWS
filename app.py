import json
import os
import sys
import boto3
import streamlit as st

# Titan Embeddings Model to generate embedding

#from langchain.embeddings import BedrockEmbeddings
#from langchain.llms.bedrock import Bedrock

from langchain_community.embeddings import BedrockEmbeddings
#from langchain_community.llms import Bedrock
from langchain_aws import ChatBedrock

# Data Ingestion

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

# LLM Model

#from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# from langchain_community.chains import RetrievalQA
#from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

# Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


# Data Ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # Split
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)

    docs=text_splitter.split_documents(documents)
    return docs

# Vector embedding and vector Store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# Claude LLM

def get_claud_llm():
    #llm=Bedrock(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",client=bedrock,model_kwargs={"max_tokens_to_sample":512})
    llm=ChatBedrock(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",client=bedrock,model_kwargs={"max_tokens":512})
    return llm

def get_llama3_llm():
    #llm=Bedrock(model_id="us.meta.llama3-1-8b-instruct-v1:0",client=bedrock,model_kwargs={"max_gen_len":512})
    llm=ChatBedrock(model_id="us.meta.llama3-1-8b-instruct-v1:0",client=bedrock,model_kwargs={"max_gen_len":512})
    return llm

prompt_template="""
Human : Use the following pieces of contextto provide a 
concise answer to the question at the end but use at least summarize with
250 words with detailed explanations. If you don't know the answer,
just say that you don't know, donc't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant

"""

PROMPT=ChatPromptTemplate.from_template(
    template=prompt_template
)

#def get_response_llm(llm,vectorstore_faiss,query):
    #qa=RetrievalQA.from_chain_type(
        #llm=llm,
        #chain_type="stuff",
        #retriever=vectorstore_faiss.as_retriever(
            #search_type="similarity",
            #search_kwargs={"k":3}
        #),
        #return_source_documents=True,
        #chain_type_kwargs={"prompt":PROMPT}
    #)
    #answer=qa({"query":query})
    #return answer["result"]

def get_response_llm(llm, vectorstore_faiss, query):

    retriever = vectorstore_faiss.as_retriever(
        search_type="similarity",
        search_kwargs={"k":3}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
    )

    response = chain.invoke(query)

    return response.content

def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with PDF using AWS Bedrock")

    user_question=st.text_input("Ask a question from the PDF file")

    with st.sidebar:
        st.title("Update or create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude output"):
        with st.spinner("Processing..."):
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_claud_llm()

            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("llama3 ouput"):
        with st.spinner("Processing..."):
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_llama3_llm()

            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__=="__main__":
    main()