from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from dotenv import load_dotenv
import hashlib
import pickle
from langchain_core.documents import Document
from langchain.tools import Tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st
DB_PATH = "vector_store"
HASH_PATH = "doc_hash.pkl"



class Retriever:
    # Constants
    def __init__(self):
        with open("C:/Users/SYED REJAUL KARIM/Downloads/Langchain_ChatBot/Langchain_ChatBot/Rag/Ares XXXI CLO Ltd.PDF", "rb") as f:
            print(f"file print ----- >{[f]}")
            current_hash =  self.compute_files_hash([f])  # ✅ wrap in list
            previous_hash = self.load_prev_hash()
            self.pages:list[Document]
            self.vectordb:FAISS
        if os.path.exists(DB_PATH) and current_hash == previous_hash:
            # Load cached FAISS DB
            self.vectordb = FAISS.load_local(DB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            loader = PyPDFLoader('C:/Users/SYED REJAUL KARIM/Downloads/Langchain_ChatBot/Langchain_ChatBot/Rag/Ares XXXI CLO Ltd.PDF')
            self.pages = loader.load()
            
        else:
            loader_Pdf=PyPDFLoader('C:/Users/SYED REJAUL KARIM/Downloads/Langchain_ChatBot/Langchain_ChatBot/Rag/Ares XXXI CLO Ltd.PDF')
            all_docs = []
            text_docs =loader_Pdf.load()
            self.pages = text_docs
            # Add metadata to each doc
            for page in text_docs:
                lines = page.page_content.splitlines()
                for line in lines:
                    if ":" in line:
                        key, value = map(str.strip, line.split(":", 1))
                        if key in ["Asset Manager", "Issuer", "Trustee","Placement Agent"]:
                            all_docs.append(Document(
                                page_content=f"{key}: {value}",
                                metadata={"section": "kv_block", "key": key}
                            ))

            # 2. Extract and chunk full body content
            splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ".", ":", " "]
            )

            for page in text_docs:
                chunks = splitter.split_text(page.page_content)
                for chunk in chunks:
                    all_docs.append(Document(
                        page_content=chunk,
                        metadata={"section": "body"}
                    ))

            embeddings = OpenAIEmbeddings(model= "text-embedding-3-small")
            vectordb = FAISS.from_documents(all_docs, embeddings)
            vectordb.save_local(DB_PATH)
            self.save_current_hash(current_hash)
        

    def retrieve(self, query, k=3):
        keywords = ["Asset Manager", "Issuer", "Trustee","Placement Agent"]
        vectordb = self.vectordb
        pages = self.pages

        # Step 1: Try exact match on known KV blocks
        if any(kw.lower() in query.lower() for kw in keywords):
            for kw in keywords:
                if kw.lower() in query.lower():
                    results = vectordb.similarity_search(query, k=5, filter={"section": "kv_block", "key": kw})
                    if results:
                        return "\n\n".join(doc.page_content for doc in results)
        
        # Step 2: Search in general paragraph text
        results = vectordb.similarity_search(query, k=5, filter={"section": "body"})
        if results:
            return "\n\n".join(doc.page_content for doc in results)
        
        # Step 3: Keyword fallback from line-by-line scan
        context = ""
        for page in pages:
            for line in page.page_content.splitlines():
                if any(k in line for k in keywords):
                    context += "\n" + line

        # Step 4: Fallback full page match
        if not context.strip():
            for page in pages:
                if query.lower() in page.page_content.lower():
                    context += "\n\n" + page.page_content

        return context.strip() if context.strip() else "No relevant data found."
    
    
    def compute_files_hash(self,files):
        sha = hashlib.sha256()
        for file in files:
            content = file.read()
            sha.update(content)
            file.seek(0)  # Reset pointer
        return sha.hexdigest()


    # Load previous hash
    def load_prev_hash(self):
        if os.path.exists(HASH_PATH):
            with open(HASH_PATH, "rb") as f:
                return pickle.load(f)
        return None


    # Save current hash
    def save_current_hash(self,hash_value):
        with open(HASH_PATH, "wb") as f:
            pickle.dump(hash_value, f)

    #-----------

   
