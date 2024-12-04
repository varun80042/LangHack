import os
import json
from typing import Dict, List, Any, Callable
import glob

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import warnings
warnings.filterwarnings("ignore")
import torch
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from openai import OpenAI

client1 = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = r"/Users/karthiknamboori/.cache/lm-studio/models/lmstudio-community/llama-3.2-3b-instruct"

os.environ["TAVILY_API_KEY"] = "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"

def get_completion(prompt, client=client1, model=model):
    """
    Obtain the response from LLM hosted by LM Studio as a server
    """
    prompt = [
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0.0,
        stream=True,
    )

    new_message = {"role": "assistant", "content": ""}

    for chunk in completion:
        if chunk.choices[0].delta.content:
            val = chunk.choices[0].delta.content
            new_message["content"] += val

    return new_message["content"]

# State for tracking conversation and context
class GraphState(BaseModel):
    query: str = Field(default="")
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    context: List[str] = Field(default_factory=list)
    result: str = Field(default="")
    retry_count: int = Field(default=0)
    route_decision: str = Field(default="")  # New field to store routing decision

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
            print(f"Loaded documents from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

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

# New node to decide routing using LLM
def decide_query_route(state: GraphState) -> Dict:
    print("Deciding Query Route...")
    
    try:
        # Load and prepare documents
        documents = load_json_files('pharmaceutical_database')  # Update path as needed
        split_docs = prepare_documents(documents)
        
        # Create vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # Perform similarity search
        similarity_results = vectorstore.similarity_search(state.query, k=3)
        
        # If relevant context is found
        if similarity_results:
            # Prepare context for relevance check
            context = [doc.page_content for doc in similarity_results]
            context_str = "\n---\n".join(context)
            print("Context avail: ", context_str)
            
            # Relevance check prompt
            relevance_prompt = f"""Evaluate if the following context is sufficient to answer the query:

            Query: {state.query}

            Context:
            {context_str}

            Respond strictly with TRUE or FALSE:"""
            
            try:
                relevance_check = get_completion(relevance_prompt).strip()
                
                # If context is deemed relevant, route to RAG
                if relevance_check == "TRUE":
                    print("Routing to RAG - Relevant context found")
                    return {
                        "route_decision": "ROUTE_RAG",
                        "context": context,  # Pass the relevant context
                        "retry_count": state.retry_count + 1
                    }
            except Exception as relevance_error:
                print(f"Relevance check error: {relevance_error}")
        
        # If no relevant context or relevance check fails
        print("Insufficient context, routing to web search")
        return {
            "route_decision": "ROUTE_WEB",
            "retry_count": state.retry_count + 1
        }
    
    except Exception as e:
        print(f"Routing decision error: {e}")
        return {
            "route_decision": "ROUTE_WEB",
            "retry_count": state.retry_count + 1
        }
# RAG Node for searching pharmaceutical database
def pharmaceutical_rag(state: GraphState) -> Dict:
    print("Running Pharmaceutical RAG...")
    
    try:
        # Load and prepare documents
        documents = load_json_files('pharmaceutical_database')  # Update path as needed
        split_docs = prepare_documents(documents)
        vectorstore = create_vector_store(split_docs)
        
        # Search for relevant documents
        results = vectorstore.similarity_search(state.query, k=5)
        
        if results:
            context = [doc.page_content for doc in results]
            return {
                "context": context,
                "result": f"Found {len(results)} relevant entries."
            }
        
        return {"context": [], "result": "No matching entries found in database."}
        
    except Exception as e:
        print(f"RAG error: {e}")
        return {"context": [], "result": "RAG search failed."}

# Web Search Node using Tavily
def web_search(state: GraphState) -> Dict:
    print("Performing Web Search...")
    try:
        tavily_client = TavilyClient(api_key="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")
        response = tavily_client.search(state.query)
        
        context = [
            f"URL: {result['url']}\nContent: {result['content']}"
            for result in response['results']
        ]
        print("Web search complete, did with: ", len(response['results'])  ," documents")
        
        return {
            "context": context,
            "result": f"Found {len(context)} web search results."
        }
    except Exception as e:
        print(f"Web search error: {e}")
        return {"context": [], "result": "Web search failed."}

# Update generate_answer to use more sophisticated prompt
def generate_answer(state: GraphState) -> Dict:
    print("Generating Answer...")
    
    context_str = "\n---\n".join(state.context)
    
    prompt_template = """You are a helpful pharmaceutical information assistant. 
    Use the following context to answer the user's query precisely and accurately.

    Query: {query}

    Context:
    {context}

    Guidelines:
    - Provide a detailed and accurate response based on the context
    - If information is insufficient, clearly state more research is needed
    - Include relevant details about the pharmaceutical topic
    - Be clear, concise, and informative

    Your detailed response:"""
    
    full_prompt = prompt_template.format(
        query=state.query,
        context=context_str
    )
    
    try:
        response = get_completion(full_prompt)
        print("Generated Answer.")
        
        return {
            "result": response,
            "chat_history": state.chat_history + [
                {"role": "user", "content": state.query},
                {"role": "ai", "content": response}
            ]
        }
    except Exception as e:
        print(f"Answer generation error: {e}")
        return {"result": "Could not generate an answer."}

# Build the Graph
def create_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("decide_route", decide_query_route)
    workflow.add_node("pharmaceutical_rag", pharmaceutical_rag)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)
    
    # Set entry point
    workflow.set_entry_point("decide_route")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "decide_route",
        lambda state: state.route_decision,
        {
            "ROUTE_RAG": "pharmaceutical_rag",
            "ROUTE_WEB": "web_search"
        }
    )
    
    # Add edges
    workflow.add_edge("pharmaceutical_rag", "generate_answer")
    workflow.add_edge("web_search", "generate_answer")
    
    # Add a final edge to end the graph
    workflow.add_edge("generate_answer", END)
    
    # Compile the graph
    return workflow.compile()

if __name__ == "__main__":
    app = create_graph()
    
    # Test queries
    queries = [
        "Summarize the details of Amoxicillin.",
        "What is the composition and primary use of Paracetamol?",
        "Can I take Ibuprofen if I have a history of stomach ulcers?",
    ]
    
    for query in queries:
        result = app.invoke({"query": query})
        print("\n" + "="*50)
        print("Question:", query)
        print("\nAnswer:", result['result'])