# 정형 축 본실험 러너 — 5장 (M1 재현 감사 + M4 표본 규모 층화)
#   데이터: NPS 국민연금 가입 사업장 내역 (59만 행, 2026-06 갱신 — 코퍼스-외부 근거는 5장 서술)
#   태스크: 회귀 — log1p(당월고지금액) ~ 시도 + 형태(법인/개인) + 업종대분류 + 사업장연령 + 가입자수
#   프로토콜(표 3.3): 기본값 프로토콜(튜닝 없음), 반복 R회(시드 상이) × 표본규모 층화, 지표 R²·MAE, 반복 분위 CI
# 사용: .venv-tab/bin/python scripts/tab_runner.py --models tabpfn,xgb,lgbm,cat --sizes 500,1000,2000,5000,10000
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "nps_사업장_raw.csv"
OUT = ROOT / "results" / "tab_nps.json"
TEST_N = 2000     # 고정 테스트셋 크기 (모든 조건 공통 — 조건 간 비교 가능성)
REPS = 5          # 반복(시드) 수 — CI는 반복 분위
SEED = 20260709


def load_nps():
    cols = ["자료생성년월", "사업장가입상태코드", "법정동주소광역시도코드", "사업장형태구분코드",
            "사업장업종코드", "적용일자", "가입자수", "당월고지금액"]
    df = pd.read_csv(RAW, encoding="cp949", usecols=lambda c: any(k in c for k in cols), low_memory=False)
    # 컬럼명을 위치 무관하게 매핑 (원명 유지 후 rename)
    ren = {}
    for c in df.columns:
        if "광역시도코드" in c: ren[c] = "시도"
        elif "형태구분" in c: ren[c] = "형태"
        elif c == "사업장업종코드": ren[c] = "업종"
        elif "적용일" in c: ren[c] = "적용일"
        elif c == "가입자수": ren[c] = "가입자수"
        elif "고지금액" in c: ren[c] = "고지금액"
        elif "상태코드" in c: ren[c] = "상태"
    df = df.rename(columns=ren)
    # 필터: 등록(1) 사업장, 유효 타깃·피처
    df = df[(df["상태"] == 1) & (df["고지금액"] > 0) & (df["가입자수"] > 0)]
    df["사업장연령"] = 2026 - pd.to_datetime(df["적용일"], errors="coerce").dt.year
    df["업종대"] = df["업종"].astype(str).str[:2]  # 고카디널리티 축약 (범주형)
    df = df.dropna(subset=["시도", "형태", "업종대", "사업장연령", "가입자수", "고지금액"])
    X = pd.DataFrame({
        "시도": df["시도"].astype("category"),
        "형태": df["형태"].astype("category"),
        "업종대": df["업종대"].astype("category"),
        "사업장연령": df["사업장연령"].astype(float),
        "가입자수": df["가입자수"].astype(float),
    })
    y = np.log1p(df["고지금액"].astype(float))
    return X.reset_index(drop=True), y.reset_index(drop=True)


def encode_for(model_key, X):
    """모델별 범주형 처리 — 동일 정보, 표현만 모델 관행에 맞춤 (기본값 프로토콜)"""
    if model_key in ("xgb", "lgbm"):
        Xe = X.copy()
        for c in Xe.select_dtypes("category"):
            Xe[c] = Xe[c].cat.codes
        return Xe
    if model_key == "cat":
        return X  # CatBoost는 cat_features 지정
    # tabpfn/tabicl: 수치 행렬 요구 → 코드화
    Xe = X.copy()
    for c in Xe.select_dtypes("category"):
        Xe[c] = Xe[c].cat.codes
    return Xe


def make_model(key):
    if key == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=0)  # 기본값
    if key == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=0, verbosity=-1)
    if key == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=0, verbose=0)
    if key == "tabpfn":
        from tabpfn import TabPFNRegressor
        # CPU >1000샘플 안전 가드 해제 (공식 플래그) — 느려질 뿐 동작은 지원됨. 런타임은 결과에 기록됨(비용 비대칭 재료)
        return TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
    if key == "tabicl":
        from tabicl import TabICLRegressor
        return TabICLRegressor(device="cpu")
    raise KeyError(key)


def run(model_keys, sizes):
    from sklearn.metrics import r2_score, mean_absolute_error
    X, y = load_nps()
    print(f"[nps] 정제 후 {len(X)}행, 피처 {list(X.columns)}")
    rng = np.random.default_rng(SEED)
    results = {}
    for mk in model_keys:
        results[mk] = {}
        for n in sizes:
            r2s, maes, t0 = [], [], time.time()
            try:
                for rep in range(REPS):
                    idx = rng.permutation(len(X))
                    te, tr = idx[:TEST_N], idx[TEST_N:TEST_N + n]
                    Xe = encode_for(mk, X)
                    model = make_model(mk)
                    if mk == "cat":
                        model.fit(X.iloc[tr], y.iloc[tr], cat_features=["시도", "형태", "업종대"])
                        pred = model.predict(X.iloc[te])
                    else:
                        model.fit(Xe.iloc[tr], y.iloc[tr])
                        pred = model.predict(Xe.iloc[te])
                    r2s.append(r2_score(y.iloc[te], pred))
                    maes.append(mean_absolute_error(y.iloc[te], pred))
                results[mk][n] = {
                    "r2_mean": float(np.mean(r2s)), "r2_ci": [float(np.quantile(r2s, .025)), float(np.quantile(r2s, .975))],
                    "mae_mean": float(np.mean(maes)), "reps": REPS, "runtime_s": round(time.time() - t0, 1),
                }
                print(f"  {mk} n={n}: R2={np.mean(r2s):.4f} [{np.quantile(r2s,.025):.4f},{np.quantile(r2s,.975):.4f}] ({results[mk][n]['runtime_s']}s)")
            except Exception as e:
                results[mk][n] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
                print(f"  {mk} n={n}: 실패 — {type(e).__name__}: {str(e)[:100]}")
                break  # 같은 모델의 더 큰 n도 실패할 것
    OUT.parent.mkdir(exist_ok=True)
    payload = {"data": "nps_사업장", "task": "회귀 log1p(고지금액)", "test_n": TEST_N, "reps": REPS, "seed": SEED,
               "protocol": "기본값(무튜닝), 표본규모 층화, 반복분위 CI", "results": results}
    if OUT.exists():
        old = json.loads(OUT.read_text())
        for mk, v in results.items():
            old.setdefault("results", {}).setdefault(mk, {}).update(v)
        old.update({k: w for k, w in payload.items() if k != "results"})
        payload = old
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--sizes", default="500,1000,2000,5000,10000")
    args = ap.parse_args()
    run(args.models.split(","), [int(s) for s in args.sizes.split(",")])
