# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:15:40 2024

@author: mahesh.joshi
"""

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
try:
    from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
    from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

import os

# Create a directory named "data" if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

st.set_page_config(page_title="Chat with the bot", page_icon="💁", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

def reset_data_folder(user_id):
    user_data_path = os.path.join("data", user_id)
    if os.path.exists(user_data_path):
        for filename in os.listdir(user_data_path):
            file_path = os.path.join(user_data_path, filename)
            os.remove(file_path)

def load_data(user_id):
    user_data_path = os.path.join("data", user_id)
    if not os.path.exists(user_data_path):
        os.makedirs(user_data_path)
    
    reader = SimpleDirectoryReader(input_dir=user_data_path, recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on data uploaded by user and your job is to answer the questions ask by user from uploaded data. Keep your answers more correct and based on facts of the uploaded data – do not hallucinate features.and also verify the answers and try to give only specific answers  based on ask question"))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

def main():
    user_id = st.session_state.get("user_id", None)
    if user_id is None:
        user_id = str(hash(str(id(st))))
        st.session_state.user_id = user_id
    
    st.title("Chat with Bot 💁")
    
    with st.sidebar:
        st.subheader("Upload Files")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
    
        if pdf_docs:
            user_data_path = os.path.join("data", user_id)
            if not os.path.exists(user_data_path):
                os.makedirs(user_data_path)
            
            for pdf in pdf_docs:
                with open(os.path.join(user_data_path, pdf.name), "wb") as f:
                    f.write(pdf.getbuffer())
            st.write("Files Uploaded!")
    
    if pdf_docs:
        st.session_state.index = load_data(user_id)  # Load data and create a new index
        
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "Ask me a question about your uploaded data"}
            ]
        
        if st.sidebar.button("Reset your uploaded data Data"):
            reset_data_folder(user_id)
            st.write("Data folder reset successful!")
        
        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.index.as_chat_engine(chat_mode="condense_question", verbose=True).chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
