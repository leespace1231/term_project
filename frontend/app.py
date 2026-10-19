"""View: Streamlit 프론트엔드"""
import base64
import os
import time

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI·데이터 분석 통합 서비스", page_icon="🤖", layout="wide")


def api(method: str, path: str, timeout: int = 120, **kwargs):
    r = requests.request(method, f"{BACKEND}{path}", timeout=timeout, **kwargs)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def show_metrics(title: str, m: dict) -> None:
    st.write(f"**{title}**")
    cols = st.columns(4)
    for col, name in zip(cols, ["recall", "f1", "auroc", "prauc"]):
        col.metric(name.upper(), m[name])
    st.write("혼동행렬 (행: 실제, 열: 예측)")
    st.dataframe(pd.DataFrame(m["confusion_matrix"], index=["실제 0", "실제 1"],
                              columns=["예측 0", "예측 1"]))


# ---------------- 사이드바 ----------------
backend_ok = False
with st.sidebar:
    st.subheader("⚙️ 시스템 상태")
    try:
        health = api("GET", "/health", timeout=3)
        backend_ok = True
        st.success("FastAPI 백엔드 연결됨")
        if health["gemini_key_set"]:
            st.info("GEMINI_API_KEY 설정 완료")
        else:
            st.error("GEMINI_API_KEY 미설정 (.env 확인)")
    except Exception:  # noqa: BLE001
        st.error(f"FastAPI 백엔드 연결 실패 ({BACKEND})")

st.title("🤖 AI·데이터 분석 통합 서비스")
if not backend_ok:
    st.warning("백엔드를 먼저 실행하세요: uvicorn backend.main:app --reload")
    st.stop()

tabs = st.tabs(["📊 댓글 대시보드", "🧪 웹훅 테스트", "📈 데이터 분석·ML",
                "🖼️ 이미지 분류·탐지", "📚 RAG Q&A", "⚙️ 시스템 설정"])


# ---------------- 1) 댓글 대시보드 ----------------
with tabs[0]:
    st.header("📋 댓글 현황 및 AI 답장 이력")
    c1, c2 = st.columns([1, 3])
    if c1.button("🔄 즉시 새로고침"):
        st.rerun()
    c2.checkbox("⏱️ 5초 자동 새로고침 (켜 두면 다른 탭 결과가 초기화될 수 있음)", key="auto")

    items = api("GET", "/comments")
    count = lambda s: sum(i["status"] == s for i in items)  # noqa: E731
    m = st.columns(4)
    m[0].metric("전체 댓글", f"{len(items)}건")
    m[1].metric("대기 중 (Pending)", f"{count('pending')}건")
    m[2].metric("완료 (Completed)", f"{count('completed')}건")
    m[3].metric("실패 (Failed)", f"{count('failed')}건")
    st.divider()

    if items:
        df = pd.DataFrame(items)
        st.dataframe(df[["comment_id", "author", "content", "status", "ai_reply", "received_at"]])
        st.subheader("🔍 댓글 상세 조회")
        sel = st.selectbox("조회할 Comment ID 선택", [i["comment_id"] for i in items])
        st.json(api("GET", f"/comments/{sel}"))
    else:
        st.info("아직 접수된 댓글이 없습니다. '웹훅 테스트' 탭에서 보내 보세요.")


# ---------------- 2) 웹훅 테스트 ----------------
with tabs[1]:
    st.header("🧪 웹훅 요청 시뮬레이터")
    st.caption("외부 플랫폼에서 수신되는 댓글 이벤트를 테스트 전송합니다.")
    st.session_state.setdefault("cid", f"cmt-{int(time.time())}")
    with st.form("webhook_form"):
        a, b = st.columns(2)
        cid = a.text_input("Comment ID *", key="cid")
        platform = b.text_input("Platform", value="test")
        pid = a.text_input("Post ID *", value="post-213")
        content = b.text_area("Content (댓글 내용) *", value="좀 더 어두운 색상으로 되나요?")
        author = a.text_input("Author *", value="user_01")
        sent = st.form_submit_button("웹훅 이벤트 전송 (POST /webhook/comment)")
    if sent:
        try:
            res = api("POST", "/webhook/comment", json={
                "comment_id": cid, "post_id": pid, "author": author,
                "content": content, "platform": platform})
            st.success("웹훅 접수 성공 (HTTP 202)")
            st.json(res)
        except Exception as e:  # noqa: BLE001
            st.error(f"전송 실패: {e}")


# ---------------- 3) 데이터 분석 · ML ----------------
with tabs[2]:
    st.header("📈 데이터 분석(EDA) 및 예측 모델")
    files = api("GET", "/data/files")["files"]

    up = st.file_uploader("CSV 업로드 (train/test 등)", type="csv", key="csv_up")
    if up and st.button("서버로 업로드"):
        try:
            api("POST", "/data/upload", files={"file": (up.name, up.getvalue(), "text/csv")})
            st.success(f"{up.name} 업로드 완료")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(e)

    if not files:
        st.info("data/ 폴더에 CSV가 없습니다. 위에서 업로드하세요.")
    else:
        train_file = st.selectbox("분석할 데이터", files)
        if st.button("EDA 실행"):
            try:
                st.session_state.eda = api("GET", "/data/eda", params={"file": train_file})
            except Exception as e:  # noqa: BLE001
                st.error(e)

        eda = st.session_state.get("eda")
        if eda and eda["file"] == train_file:
            st.write(f"**{eda['file']}** — {eda['shape'][0]}행 × {eda['shape'][1]}열")
            st.subheader("샘플 (상위 10행)")
            st.dataframe(pd.DataFrame(eda["head"]))
            st.subheader("컬럼 정보 / 결측치")
            st.dataframe(pd.DataFrame(eda["info"]))
            st.subheader("기술통계")
            st.dataframe(pd.DataFrame(eda["describe"]))
            if eda["corr"]:
                st.subheader("상관계수")
                st.dataframe(pd.DataFrame(eda["corr"]).style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))
            if eda["hist"]:
                st.subheader("수치형 분포")
                col = st.selectbox("컬럼", list(eda["hist"]))
                h = eda["hist"][col]
                labels = [f"{h['edges'][i]}~{h['edges'][i + 1]}" for i in range(len(h["counts"]))]
                st.bar_chart(pd.DataFrame({"count": h["counts"]}, index=labels))
            if eda["cat_counts"]:
                st.subheader("범주형 빈도")
                ccol = st.selectbox("범주형 컬럼", list(eda["cat_counts"]))
                st.bar_chart(pd.Series(eda["cat_counts"][ccol], name="count"))

        st.divider()
        st.subheader("🤖 후보 모델 비교 (이진 분류)")
        c1, c2 = st.columns(2)
        target = c1.text_input("타깃 컬럼", value="Survived")
        test_file = c2.selectbox("테스트 데이터 (타깃 컬럼이 있으면 최종 평가)", ["(없음)"] + files)
        if st.button("모델 학습·비교 실행"):
            with st.spinner("KNN / 로지스틱 / 나이브베이즈 / 결정트리 / 랜덤포레스트 / XGBoost 교차검증 중..."):
                try:
                    st.session_state.ml = api("POST", "/ml/run", timeout=600, json={
                        "train_file": train_file,
                        "test_file": None if test_file == "(없음)" else test_file,
                        "target": target})
                except Exception as e:  # noqa: BLE001
                    st.error(e)

        res = st.session_state.get("ml")
        if res:
            st.caption(f"제외된 컬럼: {res['dropped'] or '없음'} / 선택 기준: {res['selection_criterion']}")
            table = pd.DataFrame(res["cv_table"])
            st.write("**후보 모델 5-fold CV 성능 (recall, f1, AUROC, PRAUC)**")
            st.dataframe(table)
            st.bar_chart(table.set_index("model")[["recall", "f1", "auroc", "prauc"]], stack=False)
            st.success(f"최종 선정 모델: {res['best_model']}")
            show_metrics(f"최종 모델 홀드아웃 성과 (train의 20%, {res['holdout_rows']}행)",
                         res["holdout_metrics"])
            if res["test_metrics"]:
                show_metrics(f"최종 모델 test.csv 성과 ({res.get('test_rows')}행)", res["test_metrics"])
            if res.get("predictions"):
                pred_df = pd.DataFrame(res["predictions"])
                st.write(f"**test.csv 예측 결과 ({len(pred_df)}행)** — 정답이 없는 test.csv는 성과지표 대신 예측값을 제시")
                st.dataframe(pred_df)
                st.download_button("예측 결과 CSV 다운로드", pred_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="test_predictions.csv", mime="text/csv")
            if res["importance"]:
                st.write("**Feature Importance (상위 15)**")
                st.bar_chart(pd.DataFrame(res["importance"]).set_index("feature"))


# ---------------- 3·4) 이미지 분류 / 객체 탐지 ----------------
with tabs[3]:
    st.header("🖼️ CNN 이미지 분류 · YOLO 객체 탐지")
    img_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "webp"], key="img")
    if img_file:
        st.image(img_file.getvalue(), caption="업로드한 이미지", width=400)
        c1, c2 = st.columns(2)
        files_payload = {"file": (img_file.name, img_file.getvalue(), img_file.type or "image/jpeg")}
        if c1.button("CNN 분류 (ResNet-18)"):
            with st.spinner("분류 중..."):
                try:
                    r = api("POST", "/vision/classify", files=files_payload, timeout=300)
                    top = r["top"][0]
                    st.subheader(f"Prediction: {top['label']} ({top['prob'] * 100:.1f}%)")
                    if r["group"] != "OTHER":
                        st.write(f"→ It looks like a **{r['group']}**!")
                    st.dataframe(pd.DataFrame(r["top"]))
                except Exception as e:  # noqa: BLE001
                    st.error(e)
        conf = c2.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
        if c2.button("YOLO 객체 탐지 (YOLOv8n)"):
            with st.spinner("탐지 중..."):
                try:
                    r = api("POST", "/vision/detect", files=files_payload, params={"conf": conf}, timeout=300)
                    st.image(base64.b64decode(r["image_base64"]), caption="탐지 결과")
                    st.dataframe(pd.DataFrame(r["detections"]))
                except Exception as e:  # noqa: BLE001
                    st.error(e)


# ---------------- 5) RAG ----------------
with tabs[4]:
    st.header("📚 문서 기반 Q&A (RAG)")
    st.caption(f"현재 벡터DB 청크 수: {api('GET', '/rag/status')['chunks']}")
    doc = st.file_uploader("문서 업로드 (PDF / TXT / MD)", type=["pdf", "txt", "md"], key="doc")
    c1, c2 = st.columns(2)
    if doc and c1.button("업로드 및 임베딩"):
        with st.spinner("청킹·임베딩 중..."):
            try:
                st.success(api("POST", "/rag/upload", timeout=300,
                               files={"file": (doc.name, doc.getvalue())}))
            except Exception as e:  # noqa: BLE001
                st.error(e)
    if c2.button("벡터DB 초기화"):
        api("POST", "/rag/reset")
        st.rerun()

    question = st.text_input("질문")
    if st.button("질문하기") and question:
        with st.spinner("검색 및 답변 생성 중..."):
            try:
                r = api("POST", "/rag/ask", json={"question": question, "top_k": 4})
                st.write(r["answer"])
                with st.expander("참조한 문서 조각"):
                    for s in r["sources"]:
                        st.write("- " + s)
            except Exception as e:  # noqa: BLE001
                st.error(e)


# ---------------- 6) 시스템 설정 ----------------
with tabs[5]:
    st.header("⚙️ 시스템 설정")
    st.write(f"백엔드 URL: `{BACKEND}` (환경변수 BACKEND_URL로 변경)")
    st.json(health)
    st.markdown(
        "**구조 (MVC)**\n"
        "- Model: `backend/schemas.py`, `backend/storage.py`, `backend/services/*`\n"
        "- View: `frontend/app.py` (Streamlit)\n"
        "- Controller: `backend/main.py` (FastAPI 라우터)"
    )

# 자동 새로고침은 모든 탭 렌더링이 끝난 뒤 마지막에 실행
if st.session_state.get("auto"):
    time.sleep(5)
    st.rerun()