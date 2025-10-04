from dotenv import load_dotenv
from typing_extensions import TypedDict
from pydantic import Field,BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START,END,StateGraph
import os 
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import track
import os
import time

console = Console()
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    question:str 
    model:ChatGroq
    from_ml_topic:bool
    ai_answer:str 
    
class GradeOutput(TypedDict):
    from_ml_topic_grade:bool
    
   
   
def build_model(state:AgentState):
    model = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY,temperature=0.2)
    state["model"] = model
    return state

def get_query_topic(state:AgentState):
    print("if the question related to machine leraning.....")
    
    prompt = """
        You are a classifier. Determine if the user question is related to machine learning. 
        Given the user question, return True if it is related to machine learning, otherwise return False.
    """
    model = state["model"]
    question = state["question"]
    llm_str_output = model.with_structured_output(GradeOutput)
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",prompt),
            ("human","User question \n\n {question}"),
        ]
    )
    response = grade_prompt | llm_str_output 
    result = response.invoke({"question":question})
    state["from_ml_topic"] = result["from_ml_topic_grade"]
    return state

def grader_node(state: AgentState):
    if state['from_ml_topic']:
        return "continue"

    return "exit"

def answer_query(state: AgentState):
    print("Answering the user's query...")
    prompt = """
You are an expert in machine learning. Answer the user's query under 200 words.
    """
    model = state['model']
    query = state['question']
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", "User's Query: \n\n {query}"),
        ]
    )
    answer_chain = answer_prompt | model | StrOutputParser()
    result = answer_chain.invoke({"query": query})
    state['ai_answer'] = result

    return state

def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node(build_model, "build_model")
    workflow.add_node(get_query_topic, "get_query_topic")
    workflow.add_node(answer_query, "answer_query")

    workflow.add_edge(START, "build_model")
    workflow.add_edge("build_model", "get_query_topic")
    workflow.add_conditional_edges(
        "get_query_topic", 
        grader_node, 
        { 
            "continue": "answer_query",
            "exit": END 
        }
    )
    workflow.add_edge("answer_query", END)

    return workflow.compile()

# Query 1: "What are the latest advancements in machine learning?"
# Query 2: "What is the capital of srilanka?"

def execute_prompt_chain_workflow():
    console.print(Panel.fit("[bold magenta]✨ Machine Learning Classifier Agent ✨[/bold magenta]", border_style="magenta"))

    workflow = build_graph()  # Build once for reuse

    while True:
        question = console.input("\n[bold cyan]❓ Ask your question (or type 'exit' to quit): [/bold cyan]").strip()

        if question.lower() in ["exit", "quit", "q"]:
            console.print("\n[bold yellow]👋 Exiting the Machine Learning Classifier Agent. Goodbye![/bold yellow]\n")
            break

        if not question:
            console.print("[red]⚠️ Please enter a valid question![/red]")
            continue

        console.print(Panel.fit(f"[bold white]🧩 Processing your query...[/bold white]", border_style="cyan"))
        initial_state: AgentState = {"question": question}
        agent_response = workflow.invoke(initial_state)

        if agent_response.get("from_ml_topic"):
            console.print("\n[bold green]✅ The query was related to Machine Learning![/bold green]")
            console.print(Panel.fit(agent_response["ai_answer"], border_style="bright_green", title="🧠 ML Expert Response"))
        else:
            console.print(Panel.fit("[bold red]🚫 The query is not related to Machine Learning topics.[/bold red]", border_style="red"))

if __name__ == "__main__":
    execute_prompt_chain_workflow()
            
   
        
    
    
        
        