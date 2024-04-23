# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:48:51 2024

@author: mahesh.joshi
"""

import streamlit as st
from llama_index.llms.groq import Groq
# import openai
# from llama_index.llms.openai import OpenAI
try:
    from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
    from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

import os

# Create a directory named "data" if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")


st.set_page_config(page_title="Chat with the bot", page_icon="💁", layout="centered", initial_sidebar_state="auto", menu_items=None)
#api_key = "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"
api_key = st.secrets["Grok_Api"]
# openai.api_key = "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"
def reset_data_folder():
    for filename in os.listdir("data"):
        file_path = os.path.join("data", filename)
        os.remove(file_path)

def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    # Groq(model="mixtral-8x7b-32768", api_key="your_api_key")
    service_context = ServiceContext.from_defaults(llm=Groq(model="llama3-8b-8192", temperature=0.5, api_key=api_key, system_prompt="You are an expert on data uploaded by user and your job is to answer the questions ask by user from uploaded data. Keep your answers more correct and based on facts of the uploaded data – do not hallucinate features.and also verify the answers and try to give only specific answers  based on ask question"))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

def main():
    st.title("Chat with Bot 💁")
    
    with st.sidebar:
        st.subheader("Upload Files")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
    
        if pdf_docs:
            for pdf in pdf_docs:
                with open(os.path.join("data", pdf.name), "wb") as f:
                    f.write(pdf.getbuffer())
            st.write("Files Uploaded!")
    
    if pdf_docs:
        st.session_state.index = load_data()  # Load data and create a new index
        
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "Ask me a question about your uploaded data"}
            ]
        
        if st.sidebar.button("Reset your uploaded data Data"):
            reset_data_folder()
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
