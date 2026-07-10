# FM-Audit: 코퍼스-외부 국내 공공데이터를 이용한 파운데이션 모델 zero-shot 재현성 감사

시계열(Chronos-2·Moirai 2.0·TimesFM 2.5)·정형(TabPFN·TabICL) 파운데이션 모델의 벤치마크 성능 주장이
사전학습 코퍼스에 포함되지 않았음을 근거할 수 있는 한국 공공데이터에서 재현되는지를 감사한 학위논문의 공개 코드·결과입니다.

## 구조
- `scripts/` — 실험 러너(`ts_runner.py`, `tab_runner.py`), 데이터 로더, 그림 생성(`figures.py`)
- `results/` — 실행 결과 JSON(모델별 MASE/R²·95% CI·블록길이·런타임) + `figures/`(논문 그림 3장)
- `requirements-ts.txt` / `requirements-tab.txt` — 시계열/정형 환경 (torch 버전 충돌로 venv 분리 필요: uni2ts≤2.4 ↔ tabpfn≥2.5)

## 재현 절차
1. **환경**: Python 3.11 권장 (3.14는 scipy/torch 휠 부재). venv 2개 생성 후 각 requirements 설치. macOS는 `brew install libomp`(xgboost).
2. **데이터 취득** (공공데이터 재배포 대신 원 경로 안내 — 전처리는 로더가 수행):
   - KPX 시간별 전국 전력수요: data.go.kr 파일데이터 **15065266** (CSV, cp949) → `data/kpx_전력수요_raw.csv`
   - KEPCO 전국 시간별 전력사용량: data.go.kr **15151157** (CSV, UTF-8-sig) → `data/kepco_사용량2024_raw.csv`
   - NPS 국민연금 사업장 내역: data.go.kr **15083277** → `data/nps_사업장_raw.csv`
   - 호주 전력수요: HuggingFace `autogluon/chronos_datasets` / `monash_australian_electricity` (로더가 자동 다운로드)
3. **TabPFN 접근**: Prior Labs 계정 가입 → Licenses 탭에서 사용 버전 동의 → API 키를 `TABPFN_TOKEN` 환경변수로. CPU에서 n>1000은 `ignore_pretraining_limits=True` 필요. (이 절차 자체가 논문 5.4절 "감사 가능성 관찰"의 실측 대상이었음)
4. **실행 예**:
   ```bash
   .venv/bin/python scripts/ts_runner.py --data kpx --models snaive,chronos2,moirai2 --spread
   .venv/bin/python scripts/ts_runner.py --data kepco --models chronos2 --spread --ctx 8192   # 가용이력 층화
   .venv-tab/bin/python scripts/tab_runner.py --models tabpfn,tabicl,cat --sizes 500,1000,2000,5000,10000
   .venv/bin/python scripts/figures.py
   ```
5. **프로토콜**: rolling-origin K=8, H=168h, 이동블록 부트스트랩 B=1,000(블록길이 = 도메인 주기+ACF 컷오프 규칙, JSON에 `block_len` 기록), 정형은 5시드 분위 CI. 상세는 논문 3장 표 3.3.

## 주의
- 결과 JSON은 논문 표와 3자 대조 검증을 거친 산출물입니다. 재실행 시 포인트 추정치는 재현되어야 하며(결정적 추론), CI는 부트스트랩 시드(20260708)로 고정됩니다.
- CPU(Apple Silicon) 기준 런타임이며 GPU에서는 크게 단축됩니다.

## 라이선스
MIT (코드·결과 한정 — 원 데이터의 라이선스는 각 제공 기관의 이용약관을 따름)
