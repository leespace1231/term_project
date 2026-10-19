"""Gemini 호출 (댓글 답장, RAG 답변 공용)"""
import os

from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_client = None

REPLY_SYSTEM = (
    "당신은 브랜드 계정의 친절한 댓글 담당자입니다. "
    "댓글 작성자에게 한국어로 1~3문장, 공손하고 간결하게 답하세요. "
    "모르는 내용은 지어내지 말고 확인 후 안내하겠다고 말하세요."
)


def client() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다 (.env 확인)")
        _client = genai.Client(api_key=key)
    return _client


def generate(prompt: str, system: str | None = None) -> str:
    config = types.GenerateContentConfig(system_instruction=system) if system else None
    resp = client().models.generate_content(model=MODEL, contents=prompt, config=config)
    return (resp.text or "").strip()


def generate_reply(author: str, content: str) -> str:
    return generate(f"작성자: {author}\n댓글: {content}\n\n위 댓글에 대한 답글을 작성하세요.", REPLY_SYSTEM)
