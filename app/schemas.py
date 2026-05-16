"""Wire schemas for /chat. The spec is non-negotiable per the brief."""
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    @field_validator("content")
    @classmethod
    def _strip(cls, v: str) -> str:
        # Defensive: tolerate whitespace-only messages but normalize them.
        return v.strip() if isinstance(v, str) else v


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # SHL codes: K (knowledge), P (personality), A (ability), B (biodata), etc.


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
