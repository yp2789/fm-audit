# 양성 대조(positive control) — 측정 도구의 감도 시연 (브릿지스펙 2층·H-3 대응)
# 설계: KPX tail 8윈도에서 두 조건 비교
#   CLEAN — 정상 컨텍스트 (기존 M1과 동일)
#   LEAK  — 컨텍스트 말미 168h를 평가 대상 구간의 실제 값으로 치환 (기계적 유출: 답이 문맥에 있음)
# 해석 경계: 이는 "현실적 유출의 MDE 추정"이 아니라 "유출이 존재하면 본 계측이 반응한다"는 도구 감도의 상한 시연.
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_loaders import LOADERS
from ts_runner import MODEL_FACTORY, mase_parts, block_bootstrap_ci, acf_block_length, H, CTX, SEASON

ROOT = Path(__file__).resolve().parent.parent
series = LOADERS["kpx"]()
vals = series.to_numpy()
n = len(vals)
ends = [n - H * (i + 1) for i in range(8)][::-1]  # tail 8윈도 (ts_runner와 동일)

out = {"protocol": {"H": H, "CTX": CTX, "windows": len(ends), "design": "CLEAN vs LEAK(ctx[-168:]=target)"}, "results": {}}
for mk in ["snaive", "chronos2", "moirai2"]:
    predict = MODEL_FACTORY[mk]()
    res = {}
    for cond in ["clean", "leak"]:
        all_err, all_scale = [], []
        for e in ends:
            ctx = vals[e - CTX:e].copy()
            y = vals[e:e + H]
            if cond == "leak":
                ctx[-H:] = y  # 평가 대상 구간을 컨텍스트 말미에 그대로 주입
            yhat = np.asarray(predict(ctx, H))[:H]
            abs_err, scale = mase_parts(y, yhat, vals[:e])
            all_err.append(abs_err); all_scale.append(scale)
        pe, ps = np.concatenate(all_err), float(np.mean(all_scale))
        blk = acf_block_length(pe)
        lo, hi = block_bootstrap_ci(pe, ps, block=blk)
        res[cond] = {"mase": float(pe.mean() / ps), "ci95": [float(lo), float(hi)], "block_len": blk}
        print(f"  {mk}/{cond}: MASE={res[cond]['mase']:.4f} [{lo:.4f}, {hi:.4f}]")
    res["leak_gain_rel"] = float(1 - res["leak"]["mase"] / res["clean"]["mase"])  # 유출 조건의 상대 개선폭
    out["results"][mk] = res
    print(f"  {mk}: 유출 이득 = {out['results'][mk]['leak_gain_rel']*100:.1f}% 개선")

p = ROOT / "results" / "positive_control.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("저장:", p)
