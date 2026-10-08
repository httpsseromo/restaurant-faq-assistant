import os
import re
import threading
import time
from collections import defaultdict, deque
from typing import List, Literal

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from restaurant_bot import ask, SYSTEM_PROMPT

app = FastAPI(title="Ember & Spice Assistant")

# --- simple in-memory rate limiting ---
PER_VISITOR_LIMIT = 10      # chat messages per minute, per visitor
TOTAL_LIMIT = 300           # chat messages per hour, whole app
HANDOFF_VISITOR_LIMIT = 3   # staff alerts per hour, per visitor
HANDOFF_TOTAL_LIMIT = 20    # staff alerts per hour, whole app
visitor_hits = defaultdict(deque)
total_hits = deque()
handoff_visitor = defaultdict(deque)
handoff_total = deque()
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


def handoff_allowed(ip: str) -> bool:
    now = time.time()
    with lock:
        q = handoff_visitor[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        while handoff_total and now - handoff_total[0] > 3600:
            handoff_total.popleft()
        if len(q) >= HANDOFF_VISITOR_LIMIT or len(handoff_total) >= HANDOFF_TOTAL_LIMIT:
            return False
        q.append(now)
        handoff_total.append(now)
        return True


def clean(value, limit):
    return str(value or "").replace("\n", " ").strip()[:limit]


def digits_only(text):
    return re.sub(r"\D", "", text or "")


def valid_handoff(data, visitor_text):
    """Never trust the model's tool call: check the details in code."""
    name = clean(data.get("name"), 80)
    contact = clean(data.get("contact"), 40)
    question = clean(data.get("question"), 300)
    if not (2 <= len(name) <= 60) or not question:
        return False
    # contact must look like a phone number and nothing else
    if not re.fullmatch(r"\+?[\d\s\-().]{7,25}", contact):
        return False
    if not (7 <= len(digits_only(contact)) <= 15):
        return False
    # the details must really appear in what the visitor typed
    if name.lower() not in visitor_text.lower():
        return False
    if digits_only(contact) not in digits_only(visitor_text):
        return False
    return True


def send_handoff(data) -> bool:
    url = os.environ.get("N8N_WEBHOOK_URL")
    secret = os.environ.get("N8N_WEBHOOK_SECRET")
    if not url or not secret:
        print("[handoff skipped: n8n settings missing]")
        return False
    payload = {
        "name": clean(data.get("name"), 80),
        "contact": clean(data.get("contact"), 40),
        "question": clean(data.get("question"), 300),
    }
    try:
        r = requests.post(url, json=payload, headers={"X-Handoff-Secret": secret}, timeout=10)
        print(f"[handoff sent to n8n: status {r.status_code}]")
        return r.ok
    except Exception as e:
        print(f"[handoff failed: {type(e).__name__}]")
        return False


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
    result = ask(messages)
    if result is None:
        raise HTTPException(status_code=503, detail="Assistant is busy, please try again.")

    if result["handoff"]:
        data = result["handoff"]
        visitor_text = " ".join(m.content for m in req.messages if m.role == "user")
        if not valid_handoff(data, visitor_text):
            print("[handoff rejected: details failed validation]")
            return {"reply": "I couldn't get a valid name and phone number from that. Could you share your name and the best number to reach you on?"}
        if handoff_allowed(ip) and send_handoff(data):
            name = clean(data.get("name"), 80)
            contact = clean(data.get("contact"), 40)
            return {"reply": f"Thanks {name}! I've passed your question to our team, and they'll contact you on {contact} as soon as they can."}
        return {"reply": "Sorry, I couldn't reach our team just now. Please call us on 555-0123."}

    return {"reply": result["reply"]}


@app.get("/")
def home():
    return FileResponse("index.html")