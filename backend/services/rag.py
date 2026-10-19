"""RAG: 문서 업로드 -> 청킹 -> Gemini 임베딩 -> 벡터 검색 -> Gemini 답변 (langchain)"""
import io
import os

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import llm

_store: InMemoryVectorStore | None = None
_chunks = 0

RAG_SYSTEM = (
    "주어진 [문서 발췌]만 근거로 한국어로 답하세요. "
    "발췌에 답이 없으면 '문서에서 찾을 수 없습니다'라고 답하세요."
)


def _get_store() -> InMemoryVectorStore:
    global _store
    if _store is None:
        emb = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ["GEMINI_API_KEY"],
        )
        _store = InMemoryVectorStore(emb)
    return _store


def _extract(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    for enc in ("utf-8", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("텍스트를 디코딩할 수 없습니다.")


def add_document(filename: str, data: bytes) -> int:
    global _chunks
    text = _extract(filename, data).strip()
    if not text:
        raise ValueError("문서에서 추출된 텍스트가 없습니다.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(text)
    _get_store().add_texts(chunks, metadatas=[{"source": filename}] * len(chunks))
    _chunks += len(chunks)
    return len(chunks)


def status() -> dict:
    return {"chunks": _chunks}


def reset() -> None:
    global _store, _chunks
    _store, _chunks = None, 0


def ask(question: str, top_k: int = 4) -> tuple[str, list[str]]:
    if _chunks == 0:
        raise ValueError("업로드된 문서가 없습니다. 먼저 문서를 업로드하세요.")
    docs = _get_store().similarity_search(question, k=top_k)
    context = "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
    answer = llm.generate(f"[문서 발췌]\n{context}\n\n[질문]\n{question}", RAG_SYSTEM)
    return answer, [f"{d.metadata.get('source', '?')}: {d.page_content[:80]}..." for d in docs]
