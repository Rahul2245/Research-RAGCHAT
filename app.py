import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate




# Load Environment


load_dotenv()

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)




# Streamlit Config


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖"
)


st.title("🤖 Gemini PDF RAG Chatbot")

st.write(
    "Upload your PDF and ask questions from it"
)




# Initialize Models


@st.cache_resource
def initialize_models():


    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )


    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


    return embedding_model, llm



embedding_model, llm = initialize_models()




# Upload PDF



uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)



if uploaded_file:


    with open(
        "uploaded.pdf",
        "wb"
    ) as f:
        f.write(
            uploaded_file.getbuffer()
        )


    st.success(
        "PDF uploaded successfully"
    )



    
    # Same as Notebook
    

    @st.cache_resource
    def create_vectorstore():


        loader = PyPDFLoader(
            "uploaded.pdf"
        )


        docs = loader.load()


        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )


        chunks = splitter.split_documents(
            docs
        )


        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db"
        )


        return vectorstore



    vectorstore = create_vectorstore()



    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k":4
        }
    )



    
    # Prompt
    


    prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant for question answering using the given context.

Context:
{context}

Question:
{question}

Answer clearly and concisely based only on the context.
"""
    )



    def format_docs(docs):

        return "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )



    def rag_answer(question):

        docs = retriever.invoke(
            question
        )


        context = format_docs(
            docs
        )


        final_prompt = prompt.invoke(
            {
                "context":context,
                "question":question
            }
        )


        response = llm.invoke(
            final_prompt
        )


        return response.content



    
    # Chat UI
    


    if "messages" not in st.session_state:

        st.session_state.messages=[]



    for msg in st.session_state.messages:


        with st.chat_message(
            msg["role"]
        ):
            st.write(
                msg["content"]
            )



    question = st.chat_input(
        "Ask something about PDF..."
    )



    if question:


        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )


        with st.chat_message(
            "user"
        ):
            st.write(question)



        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                answer = rag_answer(
                    question
                )


                st.write(
                    answer
                )



        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )