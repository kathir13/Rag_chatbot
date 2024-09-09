# Run this model using streamlit run app.py --server.enableXsrfProtection false

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if API keys are set properly
groq_api_key = os.getenv("GROQ_API_KEY")
hf_key = os.getenv("HF_KEY")

if groq_api_key is None or hf_key is None:
    st.error("GROQ_API_KEY or HF_KEY is not set. Please check your .env file or environment variables.")
else:
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["HF_KEY"] = hf_key

    # Initialize the session state if not already done
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []  # Store the conversation history

    st.title("PDF Question-Answering System")
    st.write("Upload a PDF and ask questions about its content.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load PDF and split into documents
        pdf = PyPDFLoader(temp_file_path)
        load = pdf.load()
        document = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc = document.split_documents(load)

        # Initialize embeddings and vectorstore
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_key, model_name="sentence-transformers/all-MiniLM-l6-v2")
        db = Chroma.from_documents(doc, embeddings)

        # Initialize language model and prompt template
        llm = ChatGroq(model="llama3-8b-8192", temperature=1)
        prompt = ChatPromptTemplate.from_template("""
        You are an expert assistant designed to help with question-answering tasks, especially for last-minute exam preparation.
        Use the following pieces of retrieved context to answer the question as comprehensively as possible. 
        If you don't know the answer based on the given context, simply state, "I don't know." 
        Your goal is to maximize the amount of accurate information provided in your answer.
        If the user asks a question beyond the content available in the provided PDF, respond with, "The information is not specified in the PDF."

        Question: {input}

        Context: {context}""")

        # Create chains
        chain = create_stuff_documents_chain(llm, prompt=prompt)
        retriever = db.as_retriever()
        ret_chain = create_retrieval_chain(retriever, chain)
        st.write("PDF uploaded and processed successfully.")

        # Display conversation history with emojis
        for message in st.session_state.conversation_history:
            if isinstance(message, dict) and "role" in message and "content" in message:
                if message["role"] == "user":
                    st.markdown(f"**🧑‍💻 You:** {message['content']}")
                elif message["role"] == "ai":
                    st.markdown(f"**🤖 AI:** {message['content']}")
            else:
                st.warning(f"Invalid message format detected: {message}")

        # Question input
        question = st.text_input("Enter your question here:", key="input_text")

        if question:
            with st.spinner('Processing...'):
                try:
                    # Append user message to conversation history
                    st.session_state.conversation_history.append({"role": "user", "content": question})

                    # Get the answer from the chain
                    result = ret_chain.invoke({"input": question})["answer"]

                    # Append AI response to conversation history
                    st.session_state.conversation_history.append({"role": "ai", "content": result})

                    # Rerun the app to show the updated conversation
                    # st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.write("Please check the API keys and permissions.")
