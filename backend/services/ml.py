"""데이터 분석(EDA) 및 이진 분류 후보 모델 비교"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def eda(df: pd.DataFrame) -> dict:
    info = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(1).values,
        "nunique": df.nunique().values,
    })
    num = df.select_dtypes("number")
    hist = {}
    for c in num.columns:
        s = num[c].dropna()
        if s.empty:
            continue
        counts, edges = np.histogram(s, bins=10)
        hist[c] = {"counts": counts.tolist(), "edges": [round(float(e), 2) for e in edges]}
    cats = {
        c: {str(k): int(v) for k, v in df[c].value_counts().head(10).items()}
        for c in df.select_dtypes(exclude="number").columns if df[c].nunique() <= 50
    }
    describe = df.describe(include="all").T.reset_index().rename(columns={"index": "column"})
    return {
        "shape": list(df.shape),
        "head": _records(df.head(10)),
        "info": _records(info),
        "describe": _records(describe),
        "corr": json.loads(num.corr().round(3).to_json()),
        "hist": hist,
        "cat_counts": cats,
    }


def _models() -> dict:
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          eval_metric="logloss", random_state=42)
    except ImportError:
        pass
    return models


def _useless(X: pd.DataFrame) -> list[str]:
    """ID성 컬럼, 카디널리티 높은 문자열(Name, Ticket, Cabin 등) 제거"""
    drop = []
    for c in X.columns:
        if c.lower() in ("id", "passengerid"):
            drop.append(c)
        elif not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 20:
            drop.append(c)
    return drop


def _preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def _metrics(y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "auroc": round(float(roc_auc_score(y_true, proba)), 4),
        "prauc": round(float(average_precision_score(y_true, proba)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def run_classification(train: pd.DataFrame, target: str, test: pd.DataFrame | None = None) -> dict:
    if target not in train.columns:
        raise ValueError(f"타깃 컬럼 '{target}'이 없습니다. 컬럼: {list(train.columns)}")
    train = train.dropna(subset=[target])
    classes = sorted(train[target].unique().tolist())
    if len(classes) != 2:
        raise ValueError("이진 분류 타깃만 지원합니다 (클래스 2개).")
    mapping = {c: i for i, c in enumerate(classes)}
    y = train[target].map(mapping)

    X = train.drop(columns=[target])
    dropped = _useless(X)
    X = X.drop(columns=dropped)
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]

    # train을 8:2로 분할: 80%로 모델 선택(CV), 20%는 최종 모델 평가용 홀드아웃
    X_tr, X_ho, y_tr, y_ho = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"recall": "recall", "f1": "f1", "auroc": "roc_auc", "prauc": "average_precision"}
    rows, pipes = [], {}
    for name, model in _models().items():
        pipe = Pipeline([("pre", _preprocessor(num, cat)), ("model", model)])
        s = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring)
        rows.append({"model": name, **{k: round(float(np.mean(s[f"test_{k}"])), 4) for k in scoring}})
        pipes[name] = pipe

    table = pd.DataFrame(rows).sort_values(["f1", "auroc"], ascending=False).reset_index(drop=True)
    best = str(table.loc[0, "model"])
    pipe = pipes[best]
    pipe.fit(X_tr, y_tr)
    holdout_metrics = _metrics(y_ho.to_numpy(), pipe.predict_proba(X_ho)[:, 1])
    final = pipe.fit(X, y)  # 전체 train으로 재학습 (test.csv 예측용)

    result = {
        "target": target,
        "classes": classes,
        "features": X.columns.tolist(),
        "dropped": dropped,
        "cv_table": _records(table),
        "best_model": best,
        "selection_criterion": "train 80%에서 5-fold CV 평균 F1 (동률이면 AUROC)",
        "holdout_metrics": holdout_metrics,
        "holdout_rows": int(len(y_ho)),
        "test_metrics": None,
        "predictions": None,
        "importance": [],
    }

    if test is not None and target in test.columns:
        t = test.dropna(subset=[target])
        t = t[t[target].isin(classes)]
        proba = final.predict_proba(t[X.columns])[:, 1]
        result["test_metrics"] = _metrics(t[target].map(mapping).to_numpy(), proba)
        result["test_rows"] = int(len(t))

    if test is not None:
        proba_t = final.predict_proba(test[X.columns])[:, 1]
        out = pd.DataFrame({
            "predicted": [classes[i] for i in (proba_t >= 0.5).astype(int)],
            "probability": proba_t.round(4),
        })
        id_cols = [c for c in test.columns if c.lower() in ("id", "passengerid")]
        out = pd.concat([test[id_cols].reset_index(drop=True), out], axis=1)
        result["predictions"] = _records(out)

    model = final.named_steps["model"]
    names = final.named_steps["pre"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        imp = None
    if imp is not None:
        order = np.argsort(imp)[::-1][:15]
        result["importance"] = [{"feature": str(names[i]), "importance": round(float(imp[i]), 4)}
                                for i in order]
    return result