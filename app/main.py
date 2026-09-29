import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

SYSTEM_INSTRUCTION = (
    "You are a helpful customer support assistant for a small business. "
    "Keep answers concise and friendly. If you do not know something, say so "
    "rather than guessing."
)
MODEL_NAME = "gemini-3.8-flash"
MAX_MESSAGE_CHARS = 2000
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0

client = genai.Client(api_key=API_KEY)
app = FastAPI(title="Support Chatbot", version="0.1.1")

chat_sessions: dict[str, genai.chats.Chat] = {}


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    timestamp: str


def get_or_create_session(session_id: str | None) -> tuple[str, genai.chats.Chat]:
    if session_id and session_id in chat_sessions:
        return session_id, chat_sessions[session_id]
    new_id = session_id or str(uuid.uuid4())
    session = client.chats.create(
        model=MODEL_NAME,
        config={"system_instruction": SYSTEM_INSTRUCTION},
    )
    chat_sessions[new_id] = session
    return new_id, session


def send_with_retry(session: genai.chats.Chat, message: str):
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return session.send_message(message)
        except genai_errors.ServerError as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_BACKOFF_SECONDS * (2**attempt))
    raise last_error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id, session = get_or_create_session(request.session_id)

    try:
        response = send_with_retry(session, request.message)
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            raise HTTPException(
                status_code=429,
                detail="I'm receiving too many requests right now, please try again shortly.",
            ) from exc
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Model call failed after retries: {exc}"
        ) from exc

    return ChatResponse(
        session_id=session_id,
        reply=response.text,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
