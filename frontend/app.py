import os
import torch
import glob
import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def load_json_files(folder_path):
    """
    Load all JSON files from a specified folder
    """
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    all_documents = []

    for file_path in json_files:
        try:
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                text_content=False
            )

            documents = loader.load()

            for doc in documents:
                doc.metadata['source'] = file_path

            all_documents.extend(documents)

        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")

    return all_documents

def prepare_documents(documents):
    """Prepare documents by splitting them into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_vector_store(documents):
    """Create vector store for similarity search"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def setup_multi_query_retriever(vectorstore):
    """Setup multi-query retriever with custom LLM"""
    llm = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key="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"
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
        llm=llm
    )

    return retriever

def create_rag_chain1(retriever):
    """Create RAG chain for general question answering"""
    llm = ChatGroq(
        temperature=0.2,
        model_name="mixtral-8x7b-32768",
        groq_api_key="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"
    )

    prompt_template = """Use the following context from multiple documents to answer the question.
    If the answer is not in the context, admit that you don't know.

    Context: {context}

    Question: {question}

    Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def create_rag_chain2(retriever):
    """Create RAG chain for medical assistant style QA"""
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.1-8b-instant",
        groq_api_key="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"
    )

    prompt_template = """You are a medical assistant with access to knowledge about medicines.
    Use only the provided context to answer the question. If the answer cannot
    be derived from the provided context, respond with: "This information is not present in the provided documents."

    Context: {context}

    Question: {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def create_rag_chain3(retriever):
    """Create RAG chain for item summary"""
    llm = ChatGroq(
        temperature=0.2,
        model_name="mixtral-8x7b-32768",
        groq_api_key="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"
    )

    prompt_template = """
    Use the following context from multiple documents to give a summary of the item mentioned in the query. 
    The item may be directly mentioned or may be put forward as a sentence in the query. 
    Identify the apt item in the query and then generate the summary of it.
    Your task is to generate a summary only and not to handle any other task.
    If the item in the query is not in the context, admit that it is not there in context.

    Context: {context}

    Query: {question}

    Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def main():
    st.set_page_config(page_title="Medical Information RAG", page_icon="💊")
    st.title("Medical Information Retrieval Assistant")

    # Load documents and create vector store
    json_folder = 'E:\miscE\ml\LLM_Hackathon\pharmaceutical_database'  # Update this path as needed
    documents = load_json_files(json_folder)
    
    if not documents:
        st.error("No documents found. Please check the data folder.")
        return

    split_docs = prepare_documents(documents)
    vectorstore = create_vector_store(split_docs)
    retriever = setup_multi_query_retriever(vectorstore)

    # Chain selection
    chain_options = {
        "General QA": create_rag_chain1,
        "Medical Assistant Style": create_rag_chain2,
        "Item Summary": create_rag_chain3
    }

    # Sidebar for chain selection
    selected_chain = st.sidebar.selectbox(
        "Select Query Style", 
        list(chain_options.keys())
    )

    # Create the selected chain
    qa_chain = chain_options[selected_chain](retriever)

    # Query input
    query = st.text_input("Enter your medical query:", key="query_input")

    if query:
        with st.spinner("Searching and generating response..."):
            try:
                result = qa_chain({"query": query})

                # Display Answer
                st.subheader("Answer")
                st.write(result['result'])

                # Expand/Collapse Source Documents
                with st.expander("Source Documents"):
                    for i, doc in enumerate(result['source_documents'], 1):
                        st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                        st.text(doc.page_content[:500] + "...")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()