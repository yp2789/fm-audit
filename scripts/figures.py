# 핵심 결과 그림 3장 — 검증된 결과 JSON에서 직접 생성 (수치 재계산은 상대우위 %만, 4장 검증 방식과 동일)
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"   # macOS 한글
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)


def load(name):
    return json.loads((RES / name).read_text())


def stamp_source(fig, sources: str):
    """모든 그림에 출처 각주 — 본 연구 실험 산출임을 그림 자체에 명시 (외부 자료와의 혼동 방지)"""
    fig.text(0.01, 0.005, f"출처: 본 연구 실험 결과 (원자료 {sources} · 생성 experiments/scripts/figures.py)",
             fontsize=6.5, color="#777777", ha="left", va="bottom")


# ---------- 그림 1: 시계열 M1 — KPX에서 모델별 MASE (tail, 95% 블록부트스트랩 CI) ----------
kpx = load("ts_kpx.json")["results"]
order = ["chronos2", "moirai2", "timesfm", "snaive", "arima", "ets"]
labels = {"chronos2": "Chronos-2\n(유출률 0%)", "moirai2": "Moirai 2.0\n(28%)", "timesfm": "TimesFM 2.5\n(10%)",
          "snaive": "계절 naive", "arima": "AutoARIMA", "ets": "AutoETS"}
fig, ax = plt.subplots(figsize=(7, 4))
for i, k in enumerate(order):
    r = kpx[k]
    lo, hi = r["ci95"]
    c = "#2166ac" if i < 3 else "#999999"
    ax.errorbar(i, r["mase"], yerr=[[r["mase"] - lo], [hi - r["mase"]]], fmt="o", color=c, capsize=4, markersize=7)
ax.axhline(1.0, ls="--", lw=0.8, color="#cc4444")
ax.text(5.45, 1.02, "MASE=1 (계절 naive 스케일)", fontsize=8, color="#cc4444", ha="right")
ax.set_xticks(range(len(order)), [labels[k] for k in order], fontsize=9)
ax.set_ylabel("MASE (낮을수록 좋음)")
ax.set_title("그림 1. 코퍼스-외부 KPX 전력수요에서의 zero-shot 재현 (M1, tail 배치)\n파운데이션 모델 3종 모두 베이스라인과 CI 비중첩 우위 — 재현됨", fontsize=10)
fig.tight_layout()
stamp_source(fig, "results/ts_kpx.json")
fig.savefig(FIG / "fig1_시계열_M1_KPX.png")
plt.close(fig)

# ---------- 그림 2: M2 — 상대 우위 분포: KPX가 호주 분포 안에 위치 (spread) ----------
def rel_adv(res, model):
    return 100 * (1 - res[model]["mase"] / res["snaive"]["mase"])

aus = [load(f"ts_aus_elec{i}_spread.json")["results"] for i in range(5)]
kpx_sp = load("ts_kpx_spread.json")["results"]
fig, ax = plt.subplots(figsize=(7, 3.2))
for j, m in enumerate(["chronos2", "moirai2"]):
    y = 1 - j
    xs = [rel_adv(a, m) for a in aus]
    ax.scatter(xs, [y] * 5, s=60, color="#888888", alpha=0.8, label="호주 5개 주 (코퍼스-포함)" if j == 0 else None)
    ax.scatter([rel_adv(kpx_sp, m)], [y], s=130, color="#c0392b", marker="*", zorder=5, label="KPX (코퍼스-외부)" if j == 0 else None)
ax.set_yticks([1, 0], ["Chronos-2", "Moirai 2.0"])
ax.set_xlabel("계절 naive 대비 상대 우위 (%)\n유출 효과가 있다면 ●(코퍼스-포함)이 ★(외부)보다 우측에 몰려야 하나, 그런 패턴 없음")
ax.set_title("그림 2. 도메인 매칭 대조 (M2, spread 배치): 유출 신호 부재\n코퍼스-외부 KPX(★)가 코퍼스-포함 호주 주별 분포(●) 안의 평범한 위치", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(-0.6, 1.6)
fig.tight_layout()
stamp_source(fig, "results/ts_kpx_spread.json, ts_aus_elec0~4_spread.json")
fig.savefig(FIG / "fig2_M2_유출신호부재.png")
plt.close(fig)

# ---------- 그림 3: 정형 — 표본 규모에 따른 수렴 (RQ2 우위 경계) ----------
tab = load("tab_nps.json")["results"]
sizes = [500, 1000, 2000, 5000, 10000]
styles = {"tabpfn": ("#2166ac", "o", "TabPFN"), "tabicl": ("#5aa0d0", "s", "TabICL"),
          "cat": ("#b2182b", "^", "CatBoost"), "lgbm": ("#d6604d", "v", "LightGBM"), "xgb": ("#f4a582", "d", "XGBoost")}
fig, ax = plt.subplots(figsize=(7, 4.2))
for m, (c, mk, lb) in styles.items():
    ys = [tab[m][str(n)]["r2_mean"] for n in sizes]
    los = [tab[m][str(n)]["r2_ci"][0] for n in sizes]
    his = [tab[m][str(n)]["r2_ci"][1] for n in sizes]
    ax.plot(sizes, ys, marker=mk, color=c, label=lb, lw=1.5, markersize=5)
    ax.fill_between(sizes, los, his, color=c, alpha=0.12)
ax.set_xscale("log")
ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
ax.axvspan(2000, 5000, color="#f0e6a0", alpha=0.35)
ax.text(3150, 0.858, "우위 소멸 경계\n(n≈2,000~5,000)", ha="center", fontsize=8, color="#7a6a00")
ax.set_xlabel("학습 표본 규모 n (로그 스케일)")
ax.set_ylabel("R² (5시드 평균, 음영=95% 분위구간)")
ax.set_title("그림 3. 정형 축 표본 규모 층화 (M4): PFN 계열 소표본 우위 → 대표본 수렴\n코퍼스-외부 NPS 사업장 데이터에서 문헌 패턴 구조 재현", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
stamp_source(fig, "results/tab_nps.json")
fig.savefig(FIG / "fig3_정형_수렴곡선.png")
plt.close(fig)

print("생성 완료:", *[p.name for p in sorted(FIG.glob('fig*.png'))], sep="\n  ")
