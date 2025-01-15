# Import required libraries

from typing_extensions import TypedDict, List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

import os
from langgraph.graph import StateGraph, START, END
import pprint
from IPython.display import display, Markdown, Image
import gradio as gr



# Define State

class CustomerSupportState(TypedDict):
    query : str
    category : str
    sentiment : List[str]
    response : str
    
# Define a LLM model

llm = ChatGroq(
    model = "gemma2-9b-it",
    temperature= 0,
    api_key = os.getenv("GROQ_API_KEY")
)


# Create a Node function

def categorize(state: CustomerSupportState) -> CustomerSupportState:
    prompt = (
        "Categorize the following customer query into one of the following categories: Technical, Billing, or General. \n\n" f"Query: {state["query"]}\n\n"
        )
    category = llm.invoke(prompt).content.strip()
    state["category"] = category
    return state


def analyze_sentiment(state: CustomerSupportState) -> CustomerSupportState:
    prompt = (
        "Analyze the sentiment (Positive, Negative, or Neutral) of the following customer. \n\n" f"Query: {state["query"]}\n\nSentiment:",
     )
    sentiment = llm.invoke(prompt).content.strip()
    state["sentiment"] = [sentiment]
    return state

def handle_technical_query(state: CustomerSupportState) -> CustomerSupportState:
    prompt = (
        "Provide a support response to the following technical query:\n\n" f"Query:{state["query"]}\n\n"
    )
    response = llm.invoke(prompt).content.strip()
    state["response"] = response
    return state

def handle_billing_query(state: CustomerSupportState) -> CustomerSupportState:
    prompt = (
        "Provide a support response to the following billing query: \n\n" f"Query:{state["query"]}\n\n"
    )
    response = llm.invoke(prompt).content.strip()
    state["response"] = response
    return state


def handle_general_query(state: CustomerSupportState) -> CustomerSupportState:
    prompt = (
        "Provide a support response to the following general query: \n\n" f"Query:{state["query"]}\n\n"
    )
    response = llm.invoke(prompt).content.strip()
    state["response"] = response
    return state


def escalate(state: CustomerSupportState)->CustomerSupportState:
  return {"response": "This query has been escalate to a human agent due to its negative sentiment"}


def route_query(state: CustomerSupportState) -> str:
    if state["sentiment"] and state["sentiment"][0] == "Negative":
        return "escalate"
    elif state["category"] == "Technical":
        return "handle_technical_query"
    elif state["category"] == "Billing":
        return "handle_billing_query"
    else:
        return "handle_general_query"


# Create a State Graph and workflow

customer_graph = StateGraph(CustomerSupportState)

customer_graph.add_node("categorize", categorize)
customer_graph.add_node("analyze_sentiment", analyze_sentiment)
customer_graph.add_node("handle_technical_query", handle_technical_query)
customer_graph.add_node("handle_billing_query", handle_billing_query)
customer_graph.add_node("handle_general_query", handle_general_query)
customer_graph.add_node("escalate", escalate)

customer_graph.add_edge(START, "categorize")
customer_graph.add_edge("categorize", "analyze_sentiment")
customer_graph.add_conditional_edges("analyze_sentiment",
                        route_query, 
                        {
                            "handle_technical_query" : "handle_technical_query",
                            "handle_billing_query" : "handle_billing_query",
                            "handle_general_query" : "handle_general_query",
                            "escalate": "escalate",
                        }
                        )
customer_graph.add_edge("handle_technical_query", END)
customer_graph.add_edge("handle_billing_query", END)
customer_graph.add_edge("handle_general_query", END)
customer_graph.add_edge("escalate", END)

customer_support_app = customer_graph.compile()

display(Image(customer_support_app.get_graph().draw_mermaid_png()))


def test_customer_support_app(query: str):
    result = customer_support_app.invoke({"query": query})
    return {
        "category": result["category"],
        "sentiment": result["sentiment"],
        "response": result["response"]
    }
    
# Define Gradio Interface

def gradio_interface(query: str):
    result = customer_support_app.invoke({"query": query})
    # Format the result as a Markdown string
    output = (
        f"**Query:** {query}\n\n"
        f"**Category:** {result['category']}\n\n"
        f"**Sentiment:** {', '.join(result['sentiment'])}\n\n"
        f"**Response:** {result['response']}\n"
    )
    return output

    
# Build Gradion Interface

gui = gr.Interface(
    fn=gradio_interface,
    theme='Yntec/HaleyCH_Theme_Orange_Green',
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs=gr.Markdown(),
    title="Customer Support Assistant",
    description="Provide a query and receive a categorized response. The system analyzes sentiment and routes to the appropriate support channel.",
)

# Launch the app
if __name__ == "__main__":
    gui.launch(share = True)
