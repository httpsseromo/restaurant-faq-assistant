import threading
import time
from collections import defaultdict, deque
from typing import List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from restaurant_bot import ask, SYSTEM_PROMPT

app = FastAPI(title="Ember & Spice Assistant")

# --- simple in-memory rate limiting ---
PER_VISITOR_LIMIT = 10      # messages per minute, per visitor
TOTAL_LIMIT = 300           # messages per hour, whole app
visitor_hits = defaultdict(deque)
total_hits = deque()
lock = threading.Lock()


def allowed(ip: str) -> bool:
    now = time.time()
    with lock:
        q = visitor_hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        while total_hits and now - total_hits[0] > 3600:
            total_hits.popleft()
        if len(q) >= PER_VISITOR_LIMIT or len(total_hits) >= TOTAL_LIMIT:
            return False
        q.append(now)
        total_hits.append(now)
        return True


class Message(BaseModel):
    role: Literal["user", "assistant"]  # no "system": visitors can't inject rules
    content: str = Field(max_length=2000)  # bot replies count too


class ChatRequest(BaseModel):
    messages: List[Message] = Field(max_length=20)  # cap conversation length


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if not allowed(ip):
        raise HTTPException(status_code=429, detail="Too many messages, please slow down.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m.model_dump() for m in req.messages]
    reply = ask(messages)
    if reply is None:
        raise HTTPException(status_code=503, detail="Assistant is busy, please try again.")
    return {"reply": reply}


@app.get("/")
def home():
    return FileResponse("index.html")