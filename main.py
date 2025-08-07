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
    isended: str = "END"

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
    state["retrieved"] = retrieved
    return state

def grade_node(state):
    score = grader.grade(state["retrieved"])
    state["score"] = score
    return state

def generate_node(state):
    answer = generator.generate(state["retrieved"], state["query"])
    state["answer"] = answer
    print(answer)
    return state

def evaluate_node(state):
    if evaluator.is_answered(state["answer"]):
        return state
    else:
        return state

def rewrite_node(state):
    new_query = rewriter.rewrite(state["query"],state["answer"])
    return {"query": new_query}

def conditional_logic(state):
    if 1<0:
        return "endnode"
    else:
        return "rewrite"
def end_node(state):
    print("--------------------END------------------------")

# Build LangGraph
builder = StateGraph(PortfolioState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("generate", generate_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("endnode", end_node)


builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")
builder.add_edge("grade", "generate")
builder.add_edge("generate", "evaluate")
# builder.add_conditional_edges("evaluate",evaluate_node, 
#                               {"rewrite": "rewrite", "END": END}
#                             )
builder.add_conditional_edges("evaluate",conditional_logic)
builder.add_edge("rewrite", "retrieve")
#builder.add_edge("endnode", "end")
builder.set_finish_point("endnode")

# builder.set_entry_point("retrieve")
# builder.add_edge("retrieve", "generate")
# builder.add_edge("generate", END)
graph = builder.compile()
initial_state = {
    "query": "Look for Target Par Amount/Effective date par amount/Aggregate par amount",
    "retrieved": [],
    "score": 0.0,
    "answer": "",
    "isended": "END"
}
final_state = graph.invoke(initial_state)
print("\n✅ Final Answer:")
print(final_state["answer"])