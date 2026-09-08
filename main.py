import os
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

PORT = int(os.getenv("PORT", 8000))

app = FastAPI()


# --- LLM ---

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
)


# --- LangGraph ---

class State(TypedDict):
    message: str


def hello_node(state: State) -> State:
    response = llm.invoke("how's your day")
    return {"message": response.content}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("hello", hello_node)
    graph.set_entry_point("hello")
    graph.add_edge("hello", END)
    return graph.compile()


graph = build_graph()


# --- Routes ---

@app.get("/")
async def hello():
    result = await graph.ainvoke({"message": ""})
    return {"message": result["message"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
