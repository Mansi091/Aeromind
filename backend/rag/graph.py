import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from backend.rag.vector_store import retrieve_context

class AgentState(TypedDict):
    query: str
    context: str
    sources: List[str]
    response: str

def retrieve_node(state: AgentState):
    query = state["query"]
    docs = retrieve_context(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
    
    return {"context": context, "sources": sources}

def generate_node(state: AgentState):
    query = state["query"]
    context = state["context"]
    
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    
    prompt = PromptTemplate(
        template="""You are AeroMind, a professional aviation AI assistant. 
Use the following context from aviation manuals to answer the user's question accurately.
If you don't know the answer based on the context, say so. Do not hallucinate.

Context:
{context}

Question:
{query}

Answer:""",
        input_variables=["context", "query"]
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    
    return {"response": response.content}

def get_rag_graph():
    graph= StateGraph(AgentState)
    
    graph.add_node("retriever", retrieve_node)
    graph.add_node("generator", generate_node)
    
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)
    
    app = graph.compile()
    return app
