"""Model: 댓글 저장소 (JSON 파일 기반, 스레드 안전)"""
import json
import threading
from pathlib import Path
from typing import Optional

from .schemas import CommentRecord


class CommentStorage:
    def __init__(self, path: str = "data/comments.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._items: dict[str, CommentRecord] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self._items = {k: CommentRecord.model_validate(v) for k, v in raw.items()}

    def _save(self) -> None:
        data = {k: json.loads(v.model_dump_json()) for k, v in self._items.items()}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, record: CommentRecord) -> bool:
        """이미 있는 comment_id면 False (웹훅 재전송 중복 방지)"""
        with self._lock:
            if record.comment_id in self._items:
                return False
            self._items[record.comment_id] = record
            self._save()
            return True

    def get(self, comment_id: str) -> Optional[CommentRecord]:
        with self._lock:
            return self._items.get(comment_id)

    def list(self) -> list[CommentRecord]:
        with self._lock:
            return sorted(self._items.values(), key=lambda r: r.received_at, reverse=True)

    def update(self, comment_id: str, **fields) -> Optional[CommentRecord]:
        with self._lock:
            rec = self._items.get(comment_id)
            if rec is None:
                return None
            self._items[comment_id] = rec.model_copy(update=fields)
            self._save()
            return self._items[comment_id]
