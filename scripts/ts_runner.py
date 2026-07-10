# 시계열 본실험 러너 — 3장 명세 구현
#   분할: rolling-origin 홀드아웃 (K개 윈도, 표 3.3)
#   지표: MASE (스케일 = 훈련구간 계절 naive, m=24)
#   CI:   이동블록 부트스트랩 (블록 24h — 도메인 주기, B=1000, 95%)
# 사용: .venv/bin/python scripts/ts_runner.py --data kpx --models snaive,ets,chronos2 [--windows 8]
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loaders import LOADERS

ROOT = Path(__file__).resolve().parent.parent
H = 168          # 예측 지평 7일 (표 3.3)
CTX = 512        # 컨텍스트 길이 (모델 상한 내 공통값 — 표 3.3 결정 규칙)
SEASON = 24      # MASE 스케일·블록 길이의 도메인 주기
B_BOOT = 1000    # 부트스트랩 반복수
SEED = 20260708


# ---------- 모델 래퍼: 통일 인터페이스 predict(context: np.ndarray, h: int) -> np.ndarray ----------
def make_snaive():
    def predict(context, h):
        last = context[-SEASON:]
        return np.tile(last, int(np.ceil(h / SEASON)))[:h]
    return predict


def make_statsforecast(model_name):
    """AutoETS/AutoARIMA (statsforecast) — 전통 통계 베이스라인 (기본값 프로토콜: 자동 선택이 곧 기본값)"""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoARIMA

    cls = {"ets": AutoETS, "arima": AutoARIMA}[model_name]

    def predict(context, h):
        df = pd.DataFrame({"unique_id": "s", "ds": np.arange(len(context)), "y": context})
        sf = StatsForecast(models=[cls(season_length=SEASON)], freq=1, n_jobs=1)
        fc = sf.forecast(df=df, h=h)
        return fc.iloc[:, -1].to_numpy()
    return predict


def make_chronos(repo):
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", torch_dtype=torch.float32)

    def predict(context, h):
        t = torch.tensor(context, dtype=torch.float32)
        try:
            q, _ = pipe.predict_quantiles(t, prediction_length=h, quantile_levels=[0.5])
        except ValueError:
            # Chronos-2는 (n_series, n_variates, history) 3-d 입력을 요구
            q, _ = pipe.predict_quantiles(t.reshape(1, 1, -1), prediction_length=h, quantile_levels=[0.5])
        if isinstance(q, list):  # Chronos-2는 계열별 텐서 리스트 반환
            q = q[0]
        return q.reshape(-1).numpy()[:h]  # 단일 계열·단일 분위수 전제 (1,h,1)→h
    return predict


def make_moirai2(repo="Salesforce/moirai-2.0-R-small"):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    module = Moirai2Module.from_pretrained(repo)

    def predict(context, h):
        model = Moirai2Forecast(
            module=module, prediction_length=h, context_length=len(context),
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1)
        from gluonts.dataset.pandas import PandasDataset
        s = pd.Series(context, index=pd.date_range("2000-01-01", periods=len(context), freq="h"))
        ds = PandasDataset({"s": s})
        fc = next(iter(predictor.predict(ds)))
        return np.median(fc.samples, axis=0) if hasattr(fc, "samples") else fc.mean
    return predict


def make_timesfm(repo="google/timesfm-2.5-200m-pytorch"):
    # TimesFM 2.5 API (fev-bench 유출률 10% 항목과 동일 버전 — 인용 정합)
    import timesfm

    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)
    tfm.compile(timesfm.ForecastConfig(
        max_context=CTX, max_horizon=H, normalize_inputs=True,
        use_continuous_quantile_head=False,
    ))

    def predict(context, h):
        point, _ = tfm.forecast(horizon=h, inputs=[context.astype(np.float32)])
        return np.asarray(point[0])[:h]
    return predict


MODEL_FACTORY = {
    "snaive": make_snaive,
    "ets": lambda: make_statsforecast("ets"),
    "arima": lambda: make_statsforecast("arima"),
    "chronos2": lambda: make_chronos("amazon/chronos-2"),
    "chronos_bolt": lambda: make_chronos("amazon/chronos-bolt-small"),
    "moirai2": make_moirai2,
    "timesfm": make_timesfm,
}


# ---------- 평가 절차 ----------
def rolling_windows(series: pd.Series, k: int, spread: bool = False):
    """윈도 배치 (표 3.3, 시간 순서 보존):
    - 기본: 뒤에서부터 K개 연속 (꼬리 평가)
    - spread: 마지막 2년(또는 가용 구간) 안에 균등 분산 — 계절 국면 고정 교란 제거(M-4 강건성 변형)"""
    n = len(series)
    if not spread:
        ends = [n - H * (i + 1) for i in range(k)][::-1]
    else:
        span = min(n - CTX - SEASON - H, 2 * 8760)  # 최대 2년
        lo, hi = n - span, n - H
        ends = list(np.linspace(lo, hi, k).astype(int))
    return [e for e in ends if e >= CTX + SEASON]


def mase_parts(y, yhat, train_vals):
    scale = np.mean(np.abs(train_vals[SEASON:] - train_vals[:-SEASON]))
    return np.abs(y - yhat), scale


def acf_block_length(abs_err, season=SEASON, cap=7 * SEASON):
    """표 3.3 블록길이 결정 규칙 구현: 도메인 주기 + |오차| 계열 ACF 컷오프.
    블록 = max(일 주기 24, ACF가 유의대(2/√n) 아래로 처음 떨어지는 lag), 상한 = 주 주기 168.
    (H-4 수정: 고정 24 → 규칙 기반. 논문 표 3.3이 선언한 규칙을 코드가 이행)"""
    x = abs_err - abs_err.mean()
    n = len(x)
    denom = float(np.dot(x, x)) or 1.0
    thresh = 2.0 / np.sqrt(n)
    lag = season
    for k in range(1, min(cap, n - 1) + 1):
        if np.dot(x[:-k], x[k:]) / denom < thresh:
            lag = k
            break
    else:
        lag = cap
    return int(min(max(season, lag), cap))


def block_bootstrap_ci(abs_err, scale, b=B_BOOT, block=None, alpha=0.05, seed=SEED):
    """이동블록 부트스트랩 — 시간 상관 보존 재표집 (3.4 [M-1]: 교환가능성 위배 대응)
    block 미지정 시 표 3.3 규칙(acf_block_length)으로 산출."""
    if block is None:
        block = acf_block_length(abs_err)
    rng = np.random.default_rng(seed)
    n = len(abs_err)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    stats = np.empty(b)
    for i in range(b):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        sample = np.concatenate([abs_err[s:s + block] for s in starts])[:n]
        stats[i] = sample.mean() / scale
    return np.quantile(stats, [alpha / 2, 1 - alpha / 2])


def run(data_key: str, model_keys: list[str], k_windows: int, spread: bool = False, ctx: int | None = None):
    global CTX
    if ctx:
        CTX = ctx  # 가용 이력(컨텍스트) 층화: zero-shot 모델에게 계열 길이 = 활용 가능한 과거
    series = LOADERS[data_key]()
    vals = series.to_numpy()
    ends = rolling_windows(series, k_windows, spread=spread)
    mode = "spread" if spread else "tail"
    print(f"[{data_key}/{mode}] 계열 {len(series)}h, 윈도 {len(ends)}개 (H={H}, CTX={CTX})")

    results = {}
    for mk in model_keys:
        t0 = time.time()
        try:
            predict = MODEL_FACTORY[mk]()
        except Exception as e:
            print(f"  {mk}: 로드 실패 — {type(e).__name__}: {str(e)[:120]}")
            results[mk] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        all_err, all_scale, per_window = [], [], []
        for e_idx in ends:
            ctx = vals[e_idx - CTX:e_idx]
            y = vals[e_idx:e_idx + H]
            yhat = np.asarray(predict(ctx, H))[:H]
            abs_err, scale = mase_parts(y, yhat, vals[:e_idx])
            all_err.append(abs_err); all_scale.append(scale)
            per_window.append(abs_err.mean() / scale)
        pooled_err = np.concatenate(all_err)
        pooled_scale = float(np.mean(all_scale))
        mase = float(pooled_err.mean() / pooled_scale)
        blk = acf_block_length(pooled_err)
        lo, hi = block_bootstrap_ci(pooled_err, pooled_scale, block=blk)
        results[mk] = {
            "mase": mase, "ci95": [float(lo), float(hi)], "block_len": blk,
            "per_window_mase": [float(x) for x in per_window],
            "windows": len(ends), "runtime_s": round(time.time() - t0, 1),
        }
        print(f"  {mk}: MASE={mase:.4f} [{lo:.4f}, {hi:.4f}]  ({results[mk]['runtime_s']}s)")

    suffix = ("_spread" if spread else "") + (f"_ctx{CTX}" if CTX != 512 else "")
    out = ROOT / "results" / f"ts_{data_key}{suffix}.json"
    out.parent.mkdir(exist_ok=True)
    payload = {"data": data_key, "series_name": series.name, "n_hours": len(series),
               "protocol": {"H": H, "CTX": CTX, "season": SEASON, "windows": len(ends),
                            "window_mode": mode,
                            "bootstrap": {"B": B_BOOT, "block": SEASON, "ci": 0.95}, "seed": SEED},
               "results": results}
    existing = json.loads(out.read_text()) if out.exists() else payload
    existing.setdefault("results", {}).update(results)
    existing.update({k: v for k, v in payload.items() if k != "results"})
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"저장: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(LOADERS))
    ap.add_argument("--models", required=True, help="쉼표 구분: snaive,ets,arima,chronos2,chronos_bolt,moirai2,timesfm")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--spread", action="store_true", help="윈도를 최근 2년에 균등 분산 (계절 국면 교란 제거)")
    ap.add_argument("--ctx", type=int, default=None, help="컨텍스트 길이 오버라이드 (계열길이=가용이력 층화, 기본 512)")
    args = ap.parse_args()
    run(args.data, args.models.split(","), args.windows, spread=args.spread, ctx=args.ctx)
