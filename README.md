# Term-Project: AI·데이터 분석 통합 웹 애플리케이션

FastAPI(백엔드) + Streamlit(프론트엔드)로 아래 5개 모듈을 하나의 서비스로 통합했습니다.

| 모듈 | 내용 | 위치 |
|---|---|---|
| Gemini 자동 답변 | 웹훅 댓글 수신 → 백그라운드에서 Gemini 답변 생성 | `backend/main.py`, `services/llm.py` |
| 데이터 분석·ML | CSV EDA, 6개 후보 모델(KNN/LogReg/NB/DT/RF/XGBoost) 비교 | `services/ml.py` |
| CNN 분류 | ResNet-18(사전학습) 이미지 분류 | `services/vision.py` |
| YOLO 탐지 | YOLOv8n 객체 탐지 | `services/vision.py` |
| RAG Q&A | langchain 청킹 + Gemini 임베딩 + 벡터 검색 + Gemini 답변 | `services/rag.py` |

## MVC 구조
- Model: `backend/schemas.py`, `backend/storage.py`, `backend/services/*`
- View: `frontend/app.py`
- Controller: `backend/main.py`

## 실행
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # GEMINI_API_KEY 입력
# data/ 폴더에 titanic train.csv, test.csv 등을 넣거나 앱에서 업로드
uvicorn backend.main:app --reload                   # 터미널 1
streamlit run frontend/app.py                       # 터미널 2 -> http://localhost:8501
```

## Docker
```bash
docker compose up --build      # 프론트 8501, 백엔드 8000
docker build -t term-project . # 올인원 이미지 (제출용)
```

## 배포 메모
- torch + YOLO가 무거워 무료 티어(512MB RAM)에서는 메모리가 부족할 수 있음.
  Hugging Face Spaces(Docker, CPU 16GB)가 가장 무난함. 올인원 이미지는 `PORT` 환경변수(기본 8501)로 Streamlit을 띄움.
- Streamlit Community Cloud에는 프론트만 올리고 `BACKEND_URL`을 Secrets에 등록하는 방식도 가능(백엔드는 Render 등에 별도 배포).
- `.env`와 API 키는 GitHub에 올리지 말 것(.gitignore 처리됨).
