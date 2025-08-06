import os
from typing import TypedDict
from dotenv import load_dotenv
from modules.retriever import Retriever
from modules.grader import Grader
from modules.generator import Generator
from modules.rewriter import Rewriter
from modules.evaluator import Evaluator
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()
class PortfolioState(TypedDict):

    query: str

    retrieved: list

    score: float

    answer: str
# Load knowledge base
with open("data/knowledge_base.txt") as f:
    docs = f.readlines()

# retriever = Retriever(docs)
retriever = Retriever()
grader = Grader()
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")
generator = Generator(prompt)
rewriter = Rewriter()
evaluator = Evaluator()

def retrieve_node(state):
    query = state["query"]
    retrieved = retriever.retrieve(query)
    return {"query": query, "retrieved": retrieved}

# def grade_node(state):
#     score = grader.grade(state["retrieved"])
#     return {"query": state["query"], "retrieved": state["retrieved"], "score": score}

def generate_node(state):
    answer = generator.generate(state["retrieved"], state["query"])
    print(answer)
    #return {"query": state["query"], "retrieved": state["retrieved"], "score": state["score"], "answer": answer}

# def evaluate_node(state):
#     if evaluator.is_answered(state["answer"]):
#         return END
#     return "rewrite"

# def rewrite_node(state):
#     new_query = rewriter.rewrite(state["query"])
#     return {"query": new_query}

# Build LangGraph
builder = StateGraph(PortfolioState)
builder.add_node("retrieve", retrieve_node)
#builder.add_node("grade", grade_node)
builder.add_node("generate", generate_node)
#builder.add_node("evaluate", evaluate_node)
#builder.add_node("rewrite", rewrite_node)

# builder.set_entry_point("retrieve")
# builder.add_edge("retrieve", "grade")
# builder.add_edge("grade", "generate")
# builder.add_edge("generate", "evaluate")
# builder.add_conditional_edges("evaluate",evaluate_node, {"rewrite": "rewrite", END: END})
# builder.add_edge("rewrite", "retrieve")

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()
initial_state = {"query": "Effective date targer par as per the document"}
final_state = graph.invoke(initial_state)
print("\n✅ Final Answer:")
print(final_state["answer"])