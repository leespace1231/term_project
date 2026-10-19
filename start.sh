#!/bin/sh
# 올인원 실행: 백엔드(8000) + 프론트엔드(PORT, 기본 8501)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
exec streamlit run frontend/app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0
