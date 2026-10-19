"""Model: 요청/응답 스키마 (Pydantic)"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CommentStatus(str, Enum):
    pending = "pending"
    completed = "completed"
    failed = "failed"


class CommentEvent(BaseModel):
    """외부 플랫폼에서 웹훅으로 들어오는 댓글 이벤트"""
    comment_id: str
    post_id: str
    author: str
    content: str
    platform: str = "test"


class CommentRecord(CommentEvent):
    """저장되는 댓글 + AI 답변 이력"""
    status: CommentStatus = CommentStatus.pending
    ai_reply: Optional[str] = None
    received_at: datetime
    replied_at: Optional[datetime] = None
    error: Optional[str] = None


class WebhookAck(BaseModel):
    accepted: bool
    comment_id: str
    status: CommentStatus
    message: str


class MLRequest(BaseModel):
    train_file: str
    test_file: Optional[str] = None
    target: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = 4


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
