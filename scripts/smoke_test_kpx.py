# 스모크 테스트: KPX 시간별 전력수요에 Chronos-Bolt zero-shot vs 베이스라인
# 목적: 파이프라인(데이터 로드→예측→평가)이 끝까지 도는지 확인. 본 실험 아님 —
#       분할·지표는 3장 명세(rolling-origin, MASE)의 축소판만 적용.
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "kpx_전력수요_raw.csv"
OUT = ROOT / "results" / "smoke_test_kpx.md"

# --- 1. 데이터: wide(날짜×24시) → long(시간별 단일 계열) ---
df = pd.read_csv(RAW, encoding="cp949")  # 공공데이터포털 CSV는 EUC-KR 계열이 기본
hour_cols = [c for c in df.columns if c != "날짜"]
assert len(hour_cols) == 24, f"시간 컬럼 24개 기대, 실제 {len(hour_cols)}"
long = df.melt(id_vars="날짜", var_name="시", value_name="수요")
long["hour"] = long["시"].str.replace("시", "").astype(int) - 1  # 1시~24시 → 0~23
long["ts"] = pd.to_datetime(long["날짜"]) + pd.to_timedelta(long["hour"], unit="h")
series = long.sort_values("ts").set_index("ts")["수요"].astype(float)
series = series.dropna()
print(f"계열 길이: {len(series)}시간 ({series.index.min()} ~ {series.index.max()})")

# --- 2. 분할: 마지막 168시간(7일) 테스트, 직전 512시간 컨텍스트 ---
H, CTX = 168, 512
train, test = series.iloc[:-H], series.iloc[-H:]
context = train.iloc[-CTX:]

# --- 3. 베이스라인: 계절 naive (24시간 전 값) ---
naive_pred = train.iloc[-24:].values  # 마지막 하루를 7번 반복
naive_pred = np.tile(naive_pred, H // 24)

# --- 4. Chronos-Bolt zero-shot ---
import torch
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
quantiles, mean = pipe.predict_quantiles(
    torch.tensor(context.values, dtype=torch.float32),  # 이 버전 API는 inputs 위치 인자
    prediction_length=H, quantile_levels=[0.1, 0.5, 0.9],
)
chronos_pred = quantiles[0, :, 1].numpy()  # 중앙값

# --- 5. 평가: MASE (분모 = 훈련구간 계절 naive 오차) ---
def mase(y, yhat, y_train, m=24):
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return np.mean(np.abs(y - yhat)) / scale

y = test.values
rows = [
    ("Chronos-Bolt-small (zero-shot)", mase(y, chronos_pred, train.values)),
    ("계절 naive (24h)", mase(y, naive_pred, train.values)),
]

# --- 6. 결과 저장 ---
OUT.parent.mkdir(exist_ok=True)
lines = ["# 스모크 테스트 결과 — KPX 전력수요 (본 실험 아님)\n",
         f"- 계열: {len(series)}h, 테스트 {H}h, 컨텍스트 {CTX}h\n",
         "| 모델 | MASE |", "|---|---|"]
for name, s in rows:
    lines.append(f"| {name} | {s:.4f} |")
    print(f"{name}: MASE={s:.4f}")
lines.append("\nMASE<1 = 훈련구간 계절naive보다 우수. 파이프라인 관통 확인용 수치이며 논문 인용 금지.")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"저장: {OUT}")
