# 데이터 로더 — 시계열 본실험 (3장 3.5 명세)
# 반환 규격: pd.Series (DatetimeIndex, 시간별) — 러너가 동일 전처리 규율(3장 M-2 반영)로 처리
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def _common_preprocess(s: pd.Series) -> pd.Series:
    """동일 전처리 규율(3.4): 양 축(국내/해외)에 같은 규칙 적용.
    결측: 6시간 이하 간격은 선형 보간, 초과는 그대로(구간 분리는 러너 몫).
    이상치: 물리적 불가능값(<=0)만 결측 처리 — 통계적 이상치는 건드리지 않음(신호일 수 있음)."""
    s = s.astype(float)
    s[s <= 0] = np.nan
    s = s.interpolate(limit=6, limit_direction="both")
    return s


def load_kpx() -> pd.Series:
    """KPX 시간별 전국 전력수요 (코퍼스-외부, 근거등급은 모델별 — 러너 결과에 병기)"""
    raw = DATA / "kpx_전력수요_raw.csv"
    df = pd.read_csv(raw, encoding="cp949")
    hour_cols = [c for c in df.columns if c != "날짜"]
    long = df.melt(id_vars="날짜", var_name="시", value_name="수요")
    long["hour"] = long["시"].str.replace("시", "").astype(int) - 1
    long["ts"] = pd.to_datetime(long["날짜"], errors="coerce") + pd.to_timedelta(long["hour"], unit="h")
    n_bad = long["ts"].isna().sum()
    if n_bad:
        print(f"  [kpx] 날짜 파싱 실패 {n_bad}행 제거 (CSV 꼬리/요약 행)")
        long = long[long["ts"].notna()]
    s = long.sort_values("ts").set_index("ts")["수요"]
    s.name = "kpx_electricity_kr"
    s = _common_preprocess(s)
    n_nan = s.isna().sum()
    if n_nan:
        print(f"  [kpx] 보간 후 잔여 결측 {n_nan}개 제거")
        s = s.dropna()
    return s


def load_australian_electricity(state_idx: int = 0) -> pd.Series:
    """호주 전력수요 (Monash — Chronos·GiftEvalPretrain 코퍼스 포함 근거 있음 → M2 매칭 쌍의 '포함' 축).
    HF autogluon/chronos_datasets의 monash_australian_electricity 사용 (30분 → 1시간 리샘플: 동일 전처리 규율)"""
    from datasets import load_dataset

    ds = load_dataset("autogluon/chronos_datasets", "monash_australian_electricity", split="train")
    rec = ds[state_idx]
    ts = pd.to_datetime(rec["timestamp"])
    s = pd.Series(rec["target"], index=ts, name=f"australian_electricity_{rec.get('id', state_idx)}")
    # 30분 → 1시간 (KPX와 주기 통일 — 매칭 쌍의 해상도 정렬, 3.2.3 잔여 교란 축소)
    s = s.resample("1h").mean()
    return _common_preprocess(s)


LOADERS = {
    "kpx": load_kpx,
    "aus_elec": load_australian_electricity,
    # 호주 5개 주 개별 (M2 강건성 — 단일 계열 의존 제거)
    **{f"aus_elec{i}": (lambda i=i: load_australian_electricity(i)) for i in range(5)},
}


def load_kepco() -> pd.Series:
    """KEPCO 전국 시간별 전력사용량 2024 (data.go.kr 15151157, UTF-8-sig, 본부별 → 전국 합산).
    KPX 수요량과 기관·정의가 다른 별개 계열 — 제2의 국내 negative-control."""
    raw = DATA / "kepco_사용량2024_raw.csv"
    df = pd.read_csv(raw, encoding="utf-8-sig")
    df["ts"] = pd.to_datetime(df["기준일자"], errors="coerce") + pd.to_timedelta(df["기준시"].astype(int) - 1, unit="h")
    n_bad = df["ts"].isna().sum()
    if n_bad:
        print(f"  [kepco] 날짜 파싱 실패 {n_bad}행 제거")
        df = df[df["ts"].notna()]
    s = df.groupby("ts")["전력사용량"].sum().sort_index()  # 본부 합산 → 전국
    s.name = "kepco_usage_kr"
    s = _common_preprocess(s)
    n_nan = s.isna().sum()
    if n_nan:
        print(f"  [kepco] 보간 후 잔여 결측 {n_nan}개 제거")
        s = s.dropna()
    return s


LOADERS["kepco"] = load_kepco


def load_gift_electricity(item_idx: int = 0) -> pd.Series:
    """GIFT-Eval 「평가 과제」 electricity/H (브릿지스펙 §1: M1 벤치마크측 = 시험지).
    ⚠️ GiftEvalPretrain(교재)와 혼동 금지 — 이건 모델들이 성능을 보고한 평가 벤치마크다.
    단일 계열(item 0) 사용: KPX·aus_elec0과 동일한 단일계열 프로토콜 유지 (집계는 과제 변형이라 금지)."""
    from huggingface_hub import snapshot_download
    from datasets import load_from_disk

    local = snapshot_download(repo_id="Salesforce/GiftEval", repo_type="dataset",
                              allow_patterns="electricity/H/*")
    ds = load_from_disk(f"{local}/electricity/H")
    rec = ds[item_idx]
    idx = pd.date_range(rec["start"], periods=len(rec["target"]), freq="h")
    s = pd.Series(rec["target"], index=idx, name=f"gift_electricity_H_{rec['item_id']}")
    return _common_preprocess(s).dropna()


LOADERS["gift_elec"] = load_gift_electricity
