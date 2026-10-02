# MIMIC 멀티모달 30일 재입원 예측

> 흉부 X선 + EHR 결합 30일 재입원 위험 예측. 모델의 신뢰성 검증까지

성능 숫자를 올리는 것보다 "이 숫자를 믿어도 되는가"에 답하는 것을 목적으로 함.

## 요약

1. CXR 영상만으로도 재입원을 어느 정도 예측함. 자세 기준선 0.5103 대비 최고 0.5853(+0.0749).
2. 영상을 EHR 에 더해도 나아지지 않음. 파인튜닝 CNN 은 세 방식 모두 EHR 단독보다 낮고(최고 −0.0153), 동결 RAD-DINO 는
   최고 +0.0014 로 판정선(+0.02)에 못 미침.
3. 영상 단독의 신호가 EHR 에 더해도 늘지 않는 것은, 영상이 담은 정보를 EHR 이 이미 담고 있기 때문으로 보임.
4. 누수 컬럼을 빼야 숫자를 믿을 수 있음. 퇴원처(더미 8개)와 마지막 입원 여부, 누수 컬럼 9개만 넣어도 0.7011 -> 0.8271 으로
   오름. 보고한 0.7011 은 이 컬럼들을 뺀 값임.

처음에는 "영상은 자세만 본다"고 결론냈지만, 그 근거였던 모델은 학습이 되지 않은 상태였음(best epoch 1).
백본을 바꾸고 흉부 X선 파운데이션 모델로 다시 재자 판정이 뒤집힘. 경위는
[트러블슈팅 11](docs/TROUBLESHOOTING.md#11-학습이-실패한-인코더의-점수를-신호-없음으로-읽음)에 있음.

---

## 결과 요약

| Phase | 내용 | 결과 |
|---|---|---|
| 0 | 코드 뼈대 + 공개 데이터 배관 검증 | 배관 6구간 통과 |
| 1 | 코호트 정의 -> EHR 전처리 -> 환자 단위 분할 -> 정형 베이스라인 | EHR 단독 test AUROC 0.7075(문헌 ML 0.65\~0.75) |
| 1.5 | 결측 모달리티 진단 (영상 유무의 MNAR 여부) | MNAR 강함(재원일수 보정 RR 1.202, 조 RR 1.340) |
| 2 | CXR subset -> 영상 단독 모델 (백본 3종 비교) | EfficientNet +0.0534 "영상을 봤다"(시드 42, 3시드 평균 +0.0477 은 판단 보류), DenseNet121 과 ResNet50 은 판단 보류 |
| 2-B | 임베딩 캐시 추출 (이후 단계를 CPU 로) | 35,599 x 1024 |
| 2-C | RAD-DINO 동결 특징 재검증 | +0.0749, 선형 프로브로도 0.5852 |
| 3 | Fusion + M1a + V1 + V3 (인코더 2종) | 영상 단독은 신호가 있고, EHR 에 더해도 늘지 않음 |
| 4 | V4 서브그룹, 보정, SHAP, M3 DCA, 모델 비교 | 교정 후 ECE 0.0046 |
| 4-B | V5 Grad-CAM 정량화 | 균등 주의와 구분되지 않음 |
| 5 | V6 누수 A/B, V7 시간분할, M2 윈도우, 운영점, 대시보드 | 누수 부풀림 +0.1261 |
| 6 | 절제실험 8회 + 확인 실행 3시드 | 기준선을 넘은 설정 없음(승자는 기준선), test AUROC 0.5581 +/- 0.0051 |
| 부가 | 저선량 모사와 복원(소견 분류, RAD-DINO) | 1/32 까지 AUROC 하락 없음, 1/4 하락 0.0000 [-0.0002, 0.0003] |

Phase 별 결과와 해석은 [docs/report.md](docs/report.md), 저선량 부가 분석은 [docs/lowdose.md](docs/lowdose.md), 모델의
쓰임과 한계는 [docs/MODEL_CARD.md](docs/MODEL_CARD.md), TRIPOD+AI 항목별 대응은 [docs/TRIPOD_REPORT.md](docs/TRIPOD_REPORT.md) 에 있음.

## 데이터

MIMIC-IV, MIMIC-CXR-JPG 와 그 파생물(전처리 CSV, 크롭 이미지, Grad-CAM 오버레이 등)은 PhysioNet Credentialed Health
Data License 1.5.0 에 따라 저장소에 넣지 않음. PhysioNet 인증과 데이터셋별 약정 서명 뒤 `scripts/download_mimic.sh` 로
받음. 유출 차단 장치는 [docs/MODEL_CARD.md](docs/MODEL_CARD.md) 8절.

- License: https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/
- LLM 사용 공지: https://physionet.org/news/post/llm-responsible-use/

## 설치

```sh
conda env create -f environment.yml && conda activate mimic
# 또는
pip install -r requirements.txt
sh scripts/install_hooks.sh && sh scripts/verify_hooks.sh   # 데이터 유출 차단 훅(clone 마다 1회)
pytest            # 회귀 테스트 337개
```

학습은 RTX 4090(24GB, 실행 기록의 VRAM 표기 23.5GB) 한 장에서 돌림.

## 실행

데이터 위치는 저장소 기준 상대경로로 잡힘(`src/datapaths.py`). 다른 곳에
두었다면 `MIMIC_DATA_ROOT` 로 덮어씀. PhysioNet 계정명은 `PHYSIONET_USER` 로
넘기거나, 없으면 실행 시 물어봄(비밀번호는 항상 실행 시 입력하며 저장하지 않음).

```sh
export PHYSIONET_USER=본인계정            # 생략하면 실행 시 입력받음
export MIMIC_DATA_ROOT=/my/path/mimic/data  # 생략하면 저장소 기준 상대경로

sh scripts/download_mimic.sh meta        # CXR 메타데이터 (~58MB): 코호트 설계용
sh scripts/download_mimic.sh mimiciv     # MIMIC-IV v3.1 (~9.9GB)

python scripts/phase0_smoke.py           # Phase 0: 공개 데이터로 배관 검증
python scripts/run_phase1.py --check     # 데이터 준비 상태 확인
python scripts/run_phase1.py --sensitivity   # Phase 1: 코호트 + EHR 베이스라인
python scripts/run_phase1.py --no-prescriptions   # 투약 피처 없이 같은 분할로 (기여 확인)
python scripts/run_phase1_5.py               # Phase 1.5: 결측 모달리티 진단
sh scripts/download_mimic.sh images 15       # CXR JPG (목록 기준)
```

각 인자를 그렇게 고른 이유(절제로 고른 설정, `--workers`, 시드)는
[docs/report.md 재현 방법](docs/report.md#재현-방법)에 있음.

```sh
# Phase 2: 영상 단독 (GPU). 보고된 수치를 낸 설정 그대로
python scripts/run_phase2.py --workers 16 --model densenet121 --aug strong \
    --epochs 40 --patience 8 --scheduler cosine --batch 64 --size 384 \
    --weight-decay 1e-4 --freeze-epochs 3 --seed 42 --split-seed 42

# 백본 비교: 같은 조건에서 구조만 바꿈
python scripts/run_phase2.py --workers 16 --model resnet50 --aug strong \
    --epochs 40 --patience 8 --scheduler cosine --batch 64 --size 384 \
    --weight-decay 1e-4 --freeze-epochs 3 --seed 42 --split-seed 42
python scripts/run_phase2.py --workers 16 --model efficientnet_b0 --aug strong \
    --epochs 40 --patience 8 --scheduler cosine --batch 64 --size 384 \
    --weight-decay 1e-4 --freeze-epochs 3 --seed 42 --split-seed 42

python scripts/run_phase2_embed.py --workers 16 --model densenet121 --size 384
python scripts/run_phase2_raddino.py --workers 16   # Phase 2-C: RAD-DINO 동결 특징
```

```sh
# 절제실험 8회 + 기준선 시드 반복 2회. 회차별 인자와 가설은 docs/EXPERIMENT_LOG.md
python scripts/run_ablation.py --workers 16 --id 1 --name baseline \
    --aug strong --batch 64 --size 384 --epochs 40 --patience 8 \
    --weight-decay 1e-4 --freeze-epochs 3 --seed 42 \
    --hypothesis "기준선. 384px 와 batch 64 를 출발점으로 삼는다" --change "없음(기준선)"
# ... 나머지 9회는 EXPERIMENT_LOG 참고
python scripts/render_experiment_log.py      # 기록 -> docs/EXPERIMENT_LOG.md
```

```sh
# 확인 실행: 절제 승자(기준선) 설정으로 test 를 시드 셋에서 한 번씩
for s in 42 43 44; do
  t=ablated; [ "$s" = 42 ] || t=ablated_s$s    # 결과 파일 phase2_metrics_ablated{,_s43,_s44}.json
  python scripts/run_phase2.py --workers 16 --model efficientnet_b0 \
      --tag "$t" --aug strong --epochs 40 --patience 8 \
      --scheduler cosine --batch 64 --size 384 --weight-decay 1e-4 \
      --freeze-epochs 3 --seed $s --split-seed 42
done
```

```sh
# 이하 CPU. 임베딩 캐시를 읽음
python scripts/run_phase3.py                 # Fusion, M1a, V1, V3
python scripts/run_phase3.py --encoder raddino     # 인코더 바꿔 재검증
python scripts/run_phase4.py                 # 보정, SHAP, DCA, 서브그룹
python scripts/run_phase4_gradcam.py --model densenet121 --size 384   # Phase 4-B (GPU)
python scripts/run_phase4_gradcam.py --model efficientnet_b0 --size 384
python scripts/run_phase5.py                 # 누수 A/B, 시간분할, 창, 운영점
python scripts/run_phase5.py --encoder raddino
```

## 더 읽을 것

[docs/report.md](docs/report.md) 에 설계 원칙(예측 시점, 누수 차단 컬럼, 지표 규칙), 성능 기대치(문헌 성능과
코호트 기준별 재입원률), 재현 방법과 코드 구조가 있음.

## 라이선스

코드는 MIT. 데이터는 PhysioNet DUA 를 따르며 이 저장소에 포함되지 않음. [LICENSE](LICENSE) 참조.

## 트러블슈팅

실측으로 잡은 결함 26건의 경위, 수정, 회귀 테스트는 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
에 있음. 번호는 문서들이 가리키는 번호 그대로임. "(판정이 바뀜)" 은 고친 뒤 결과 판정이 뒤집힌 항목임.

1. 점이연 상관만 보면 오판함(판정이 바뀜)
2. 그 위험비가 재원일수로 교란돼 있었음
3. 입력에 섞여 있던 문제 컬럼 3건
4. `admit_year` 는 쓸 수 없음. MIMIC-IV 날짜 시프트
5. 우측 절단. 마지막 입원 38.9% 가 무조건 음성이 됨
6. 데이터 품질. `dischtime <= admittime` 오류 180건
7. 다운로드가 고른 파일과 학습이 고른 파일이 달랐음. 코호트 18.6% 소실
8. 측면 오염을 걷어내고 다시 학습함. 당시 판정은 그대로였음
9. Phase 3 코드에서 잡은 결함 4건
10. Phase 4, 5 에서 잡은 결함 4건. 둘은 결론의 의미를 바꿈
11. 학습이 실패한 인코더의 점수를 "신호 없음"으로 읽음(판정이 바뀜)
12. M2 는 촬영 시점이 아니라 데이터 양을 재고 있었음
13. 12에폭 한도가 조기종료 역할을 하고 있었음. 에폭을 늘리자 과적합이 늘었음
14. 설정을 바꿔가며 추적한 기록이 없었음
15. AUROC 변동폭으로 PR-AUC 차이를 판정하려 함
16. 시드를 고정했는데도 두 스크립트의 학습 결과가 달랐음
17. 새 설정으로 학습해 놓고 임베딩은 옛 체크포인트에서 뽑고 있었음
18. 절제실험의 차이가 시드 하나 바꾼 폭보다 작았음
19. 라벨을 보고 학습한 인코더의 임베딩을, 그 인코더가 본 환자에게 다시 씀(판정이 바뀜)
20. 영상이 없다는 사실이 학습과 평가에서 다른 모양이었음
21. cosine 스케줄러의 하한이 한쪽 파라미터 그룹 기준으로만 걸림
22. 확인 실행이 절제 승자가 아닌 설정을 검증하고 있었음
23. GBDT 의 시드가 어디에도 적혀 있지 않았음
24. 재입원 라벨의 다음 입원을 걸러진 코호트 안에서만 찾고 있었음(판정이 바뀜)
25. Charlson 신장 질환의 ICD-9 코드가 넓게 잡혀 있었음
26. DnCNN 한 시드가 아무것도 배우지 못함
