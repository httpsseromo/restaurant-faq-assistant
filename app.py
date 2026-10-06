from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from restaurant_bot import ask, SYSTEM_PROMPT

app = FastAPI(title="Ember & Spice Assistant")


class Message(BaseModel):
    role: Literal["user", "assistant"]  # no "system": visitors can't inject rules
    content: str = Field(max_length=2000)  # bot replies count too


class ChatRequest(BaseModel):
    messages: List[Message] = Field(max_length=20)  # cap conversation length


@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m.model_dump() for m in req.messages]
    reply = ask(messages)
    if reply is None:
        raise HTTPException(status_code=503, detail="Assistant is busy, please try again.")
    return {"reply": reply}


@app.get("/")
def home():
    return FileResponse("index.html")