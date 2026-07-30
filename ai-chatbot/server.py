"""
🤖 AI Chatbot — FastAPI REST Server
=====================================
A REST API that exposes the chatbot for HTTP clients.

Usage:
    uvicorn server:app --reload --port 8000

Endpoints:
    POST /chat          — Send a message and get a response
    GET  /health        — Health check
    GET  /history       — Get chat history for current session
    GET  /intents       — List available intent tags
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import ChatBot

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Chatbot API",
    description="An NLP-powered chatbot REST API using TF-IDF and cosine similarity.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot (shared across requests)
bot = ChatBot()


# ── Models ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""
    response: str
    confidence: float
    intent_matched: bool


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    intents_loaded: int
    patterns_loaded: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the chatbot and receive a response."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    response_text, confidence = bot.get_response(request.message)

    # Log to history
    bot.chat(request.message)

    return ChatResponse(
        response=response_text,
        confidence=round(confidence, 4),
        intent_matched=confidence >= bot.confidence_threshold,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check the health status of the chatbot API."""
    return HealthResponse(
        status="healthy",
        intents_loaded=len(bot.knowledge_base.get("intents", [])),
        patterns_loaded=len(bot.all_patterns),
    )


@app.get("/history")
async def history():
    """Get the chat history for the current session."""
    return {"history": bot.get_history(), "total_messages": len(bot.get_history())}


@app.get("/intents")
async def intents():
    """List all available intent tags in the knowledge base."""
    tags = [
        intent["tag"]
        for intent in bot.knowledge_base.get("intents", [])
    ]
    return {"intents": tags, "total": len(tags)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
