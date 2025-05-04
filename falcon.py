import streamlit as st
from langchain_community.document_loaders import TextLoader
from pypdf import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


def read_pdf(file):
    document = ""
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
    return document


def read_txt(file):
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")
    return document


def split_doc(document, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)
    return split


def embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name):
    instructor_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    db = FAISS.from_documents(split, instructor_embeddings)

    if create_new_vs:
        db.save_local("vector store/" + new_vs_name)
    else:
        load_db = FAISS.load_local(
            "vector store/" + existing_vector_store,
            instructor_embeddings,
            allow_dangerous_deserialization=True
        )
        load_db.merge_from(db)
        load_db.save_local("vector store/" + new_vs_name)

    st.success("The document has been saved.")


def prepare_rag_llm(api_key, vector_store_list, temperature, max_length):
    llm = ChatGroq(
        api_key=api_key,
        model="llama3-8b-8192",
        temperature=temperature,
        max_tokens=max_length
    )

    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Case 1: No vector store → Just a chatbot (no RAG)
    if vector_store_list == "None":
        from langchain.chains import ConversationChain

        return ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )

    # Case 2: Vector store selected → Use RAG
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    loaded_db = FAISS.load_local(
        f"vector store/{vector_store_list}",
        embeddings,
        allow_dangerous_deserialization=True
    )

    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory,
    )

    return qa_conversation


def generate_answer(question, api_key):
    if not api_key:
        return "Insert the Groq API key", ["no source"]

    response = st.session_state.conversation({"question": question})

    if isinstance(response, dict) and "answer" in response:
        answer = response["answer"].split("Helpful Answer:")[-1].strip()
        doc_source = [doc.page_content for doc in response.get("source_documents", [])]
    else:
        answer = response["response"] if isinstance(response, dict) else str(response)
        doc_source = ["No sources used"]

    return answer, doc_source
