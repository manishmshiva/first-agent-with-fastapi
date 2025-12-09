from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()

# defining the tool that LLM can call
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Creating an agent
agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "Welcome to your first agent"}

@app.post("/chat")
def chat(request: ChatRequest):
    result = agent.invoke({"messages":[{"role":"user","content":request.message}]})
    return {"reply": result["messages"][-1].content}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)