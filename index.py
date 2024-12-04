import os
import json
from typing import Dict, List, Any, Callable

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import os
import json
from typing import Dict, List, Any, Callable

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

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


# Load pharmaceutical data (you'll need to replace this with your actual JSON data)
def load_pharmaceutical_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading pharmaceutical data: {e}")
        return {}

# RAG Node for searching pharmaceutical database
def pharmaceutical_rag(state: GraphState) -> Dict:
    print("Running Pharmaceutical RAG...")
    
    # Load your pharmaceutical database (replace with actual path)
    pharma_data = load_pharmaceutical_data('pharmaceutical_database.json')
    
    # Simple search through pharmaceutical data
    matching_entries = []
    for entry in pharma_data:
        if state.query.lower() in json.dumps(entry).lower():
            matching_entries.append(entry)
    
    if matching_entries:
        context = [json.dumps(entry) for entry in matching_entries]
        return {
            "context": context,
            "result": f"Found {len(matching_entries)} matching entries."
        }
    
    return {"context": [], "result": "No matching entries found in database."}

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
    
# LLM Node for generating final answer
def generate_answer(state: GraphState) -> Dict:
    print("Generating Answer...")
    
    # Combine context and create prompt
    context_str = "\n---\n".join(state.context)
    
    # Construct a comprehensive prompt
    full_prompt = f"""You are a helpful pharmaceutical information assistant. 
    Use the following context to answer the user's query precisely and accurately.

    Query: {state.query}

    Context:
    {context_str}

    Guidelines:
    - Provide a detailed and accurate response based on the context
    - If information is insufficient, clearly state more research is needed
    - Include relevant details about the pharmaceutical topic
    - Be clear, concise, and informative

    Your detailed response:"""
    
    try:
        # Use the get_completion function
        response = get_completion(full_prompt)
        
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

# Router Node to decide next action
def route_query(state: GraphState) -> str:
    print("Routing Query...")
    
    # If retry count is high, end the process
    if state.retry_count >= 2:
        return END
    
    # Check if pharmaceutical RAG provided a good answer
    if state.context and "No matching entries" not in state.result:
        print("Valid chunks found in the rag, performing generation using rag chunks...")
        return "generate_answer"
    
    # Otherwise, try web search
    print("NOthing found in rag, performing web search")
    return "web_search"

# Build the Graph
def create_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("pharmaceutical_rag", pharmaceutical_rag)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)
    
    # Set entry point
    workflow.set_entry_point("pharmaceutical_rag")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "pharmaceutical_rag",
        route_query,
        {
            "web_search": "web_search",
            "generate_answer": "generate_answer",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "web_search",
        route_query,
        {
            "web_search": "web_search",
            "generate_answer": "generate_answer",
            END: END
        }
    )
    
    # Compile the graph
    return workflow.compile()

app = create_graph()
queries = [
"What is the composition and primary use of Paracetamol?",
"Can I take Ibuprofen if I have a history of stomach ulcers?"
]

for query in queries:
    result = app.invoke({"query": query})
    print(result)


print('hi')