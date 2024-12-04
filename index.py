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
client1 = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")  # html to json
model = r"/Users/karthiknamboori/.cache/lm-studio/models/lmstudio-community/llama-3.2-3b-instruct"

os.environ["TAVILY_API_KEY"] = "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 "

def get_completion(prompt, client=client1, model=model):
    """
    given the prompt, obtain the response from LLM hosted by LM Studio as a server
    :param prompt: prompt to be sent to LLM server
    :return: response from the LLM
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
            # print(chunk.choices[0].delta.content, end="", flush=True)
            val = chunk.choices[0].delta.content
            new_message["content"] += val

    # print(type()
    val = new_message["content"]  # .split("<end_of_turn>")[0]

    return val

prompt = """
    You are a political leader and your party is trying to win the general elections in India. 
    You are given an LLM that can provide you the analytics using the past historical data given to it.
    In particular the LLM has been provided data on which party won each constituency out of 545 and which assembly segment within the main constituency is more favorable.
    It also has details of votes polled by every candidate.
    Tell me 10 questions that you want to ask the LLM.
    """
# results = get_completion(prompt)

# State for tracking conversation and context
class GraphState(BaseModel):
    query: str = Field(default="")
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    context: List[str] = Field(default_factory=list)
    result: str = Field(default="")
    retry_count: int = Field(default=0)

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

# Load pharmaceutical data (you'll need to replace this with your actual JSON data)
def load_pharmaceutical_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading pharmaceutical data: {e}")
        return {}

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
        
        return {
            "context": context,
            "result": f"Found {len(context)} web search results."
        }
    except Exception as e:
        print(f"Web search error: {e}")
        return {"context": [], "result": "Web search failed."}


# test web search
def test_web_search(query):
    print("Performing Web Search...")
    try:
        tool = TavilySearchResults(max_results=3)
        search_results = tool.invoke("What's a 'node' in LangGraph?")
        
        context = [
            f"URL: {result['url']}\nContent: {result['content']}"
            for result in search_results
        ]
        
        return {
            "context": context,
            "result": f"Found {len(context)} web search results."
        }
    except Exception as e:
        print(f"Web search error: {e}")
        return {"context": [], "result": "Web search failed."}
    
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_system.log'),
        logging.StreamHandler()
    ]
)

def generate_rag_response(context: str, query: str) -> str:
    """Generate RAG-based response using context"""
    logging.info(f"Generating RAG response for query: {query}")
    logging.debug(f"Context provided: {context}")
    
    prompt_template = """
    You are a helpful assistant. Use ONLY the following context to answer the user's query.
    If the context doesn't contain relevant information to answer the query, you must indicate that.
    
    Context:
    {context}
    
    Query: {query}
    
    Instructions:
    1. Only use information from the provided context
    2. If the context doesn't contain relevant information, explicitly state 'No relevant information found'
    3. Be specific and precise in your response
    4. Cite specific information from the context when possible
    
    Response:"""
    
    prompt = prompt_template.format(context=context, query=query)
    response = get_completion(prompt)
    logging.info(f"RAG response generated: {response}")
    return response

def generate_web_search_response(context: str, query: str) -> str:
    """Generate response using web search results"""
    logging.info(f"Generating web search response for query: {query}")
    logging.debug(f"Web search context: {context}")
    
    prompt_template = """
    Use the following web search results to answer the user's query.
    
    Web Search Results:
    {context}
    
    Query: {query}
    
    Instructions:
    1. Use the web search results to provide a comprehensive answer
    2. If no relevant information is found, explicitly state that
    3. Be specific and cite sources when possible
    
    Response:"""
    
    prompt = prompt_template.format(context=context, query=query)
    response = get_completion(prompt)
    logging.info(f"Web search response generated: {response}")
    return response

def generate_json_response(raw_response: str) -> Dict:
    """Convert raw response to structured JSON format"""
    logging.info("Converting raw response to JSON format")
    logging.debug(f"Raw response: {raw_response}")
    
    prompt_template = """
    Analyze the following response and convert it into a JSON format that indicates whether it contains relevant information.
    If the response indicates it couldn't answer the query or lacks relevant information, set relevantResponseGenerated to false.
    
    Response to analyze:
    {raw_response}
    
    Output the response in this exact JSON format:
    {{
        "relevantResponseGenerated": boolean (True or False),
        "relevantResponse": "detailed response here if relevant, empty string if not",
        "source": "rag or web_search"
    }}
    
    Only output valid JSON, no additional text."""
    
    prompt = prompt_template.format(raw_response=raw_response)
    json_response = get_completion(prompt)
    logging.info(f"JSON response generated: {json_response}")
    
    try:
        return json.loads(json_response)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return {
            "relevantResponseGenerated": False,
            "relevantResponse": "",
            "source": "error"
        }

def generate_answer(state: GraphState) -> Dict:
    logging.info(f"Generating answer for query: {state.query}")
    
    # First try RAG
    context_str = "\n---\n".join(state.context)
    raw_rag_response = generate_rag_response(context_str, state.query)
    print("Raw RAG -> ", raw_rag_response)
    json_rag_response = generate_json_response(raw_rag_response)
    print("JSON response -> ", json_rag_response)
    
    # If RAG found relevant information, return it
    if json_rag_response.get("relevantResponseGenerated", False):
        logging.info("Relevant RAG response found")
        return {
            "result": json_rag_response["relevantResponse"],
            "chat_history": state.chat_history + [
                {"role": "user", "content": state.query},
                {"role": "ai", "content": json_rag_response["relevantResponse"]}
            ],
            "source": "rag"
        }
    
    # If RAG didn't find relevant info, try web search
    logging.info("No relevant RAG response, trying web search")
    web_search_results = web_search(state)
    web_context = "\n---\n".join(web_search_results.get("context", []))
    
    raw_web_response = generate_web_search_response(web_context, state.query)
    json_web_response = generate_json_response(raw_web_response)
    
    if json_web_response.get("relevantResponseGenerated", False):
        logging.info("Relevant web search response found")
        return {
            "result": json_web_response["relevantResponse"],
            "chat_history": state.chat_history + [
                {"role": "user", "content": state.query},
                {"role": "ai", "content": json_web_response["relevantResponse"]}
            ],
            "source": "web_search"
        }
    
    # If neither source found relevant info
    logging.warning("No relevant information found from either RAG or web search")
    return {
        "result": "I apologize, but I couldn't find relevant information to answer your query from either our database or web search.",
        "chat_history": state.chat_history + [
            {"role": "user", "content": state.query}
        ],
        "source": "none"
    }

# Router Node to decide next action
def route_query(state: GraphState) -> str:
    logging.info("Routing query...")
    
    # If we have a final result, end the process
    if state.result and state.result != "No matching entries":
        logging.info("Valid result found, ending process")
        return END
    
    # If we haven't tried RAG yet, try it first
    if not state.context:
        logging.info("No context yet, routing to RAG")
        return "pharmaceutical_rag"
    
    # If RAG didn't find relevant info, try web search
    if state.result == "No matching entries" and state.source != "web_search":
        logging.info("RAG found no matches, routing to web search")
        return "web_search"
    
    # If we've tried both RAG and web search, end the process
    logging.info("All sources exhausted, ending process")
    return END
# Build the Graph
def create_graph():
    logging.info("Creating workflow graph")
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("pharmaceutical_rag", pharmaceutical_rag)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)
    
    # Set entry point
    workflow.set_entry_point("pharmaceutical_rag")
    
    # Add edges from RAG
    workflow.add_conditional_edges(
        "pharmaceutical_rag",
        route_query,
        {
            "generate_answer": "generate_answer",
            "web_search": "web_search",
            END: END
        }
    )
    
    # Add edges from web search
    workflow.add_conditional_edges(
        "web_search",
        route_query,
        {
            "generate_answer": "generate_answer",
            END: END
        }
    )
    
    # Add edges from answer generation
    workflow.add_conditional_edges(
        "generate_answer",
        route_query,
        {
            "web_search": "web_search",
            END: END
        }
    )
    
    logging.info("Compiling workflow graph")
    return workflow.compile()

# class GraphState(BaseModel):
#     query: str = Field(default="")
#     chat_history: List[Dict[str, str]] = Field(default_factory=list)
#     context: List[str] = Field(default_factory=list)
#     result: str = Field(default="")
#     retry_count: int = Field(default=0)
#     source: str = Field(default="")  # Added to track source of information

# Update the main execution with better logging
if __name__ == "__main__":
    logging.info("Initializing QA System")
    app = create_graph()
    
    queries = [
        "What is the composition and primary use of Paracetamol?",
        "Can I take Ibuprofen if I have a history of stomach ulcers?",
        "Summarize the details of Amoxicillin."
    ]
    
    for query in queries:
        logging.info(f"\n{'='*50}\nProcessing new query: {query}")
        
        try:
            result = app.invoke({
                "query": query,
                "chat_history": [],
                "context": [],
                "result": "",
                "retry_count": 0,
                "source": ""
            })
            
            print("\n" + "="*50)
            print("Question:", query)
            print("Source:", result.get('source', 'unknown'))
            print("\nAnswer:", result['result'])
            
            logging.info(f"Query completed. Source: {result.get('source', 'unknown')}")
            logging.debug(f"Full result: {result}")
            
        except Exception as e:
            logging.error(f"Error processing query '{query}': {str(e)}", exc_info=True)
            print(f"Error processing query: {str(e)}")