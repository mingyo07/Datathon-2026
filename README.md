# Datathon 2026

2026년 11월 14일~15일 데이터톤 참가를 위한 팀 프로젝트입니다.

## 1. 프로젝트 목표

주어진 데이터를 분석하고 머신러닝 모델을 개발하여
예측 성능을 개선하는 것을 목표로 합니다.

핵심 전략:
- 데이터에 대한 도메인 지식 확보
- 데이터 품질 분석 및 적절한 전처리
- 도메인 기반 파생변수 생성
- 다양한 ML 모델 성능 비교
- 실험 결과 기록 및 개선

※ 구체적인 문제, Target, 평가 지표는 대회 공개 후 작성합니다.

## 2. 팀원 역할

| 담당 | 주요 업무 |
|---|---|
| member1 | EDA 및 데이터 품질 분석 |
| member2 | Baseline 모델 개발 및 성능 평가 |
| member3 | 도메인 분석 및 파생변수 생성 |

역할은 주담당을 의미하며, 모든 팀원은 다른 작업을 지원할 수 있습니다.

## 3. 기술 스택

- Python
- Pandas / NumPy
- Scikit-learn
- LightGBM / XGBoost / CatBoost (필요 시)
- Google Colab
- Google Drive
- GitHub

## 4. 폴더 구조

Datathon-2026/
├── data/
│   ├── raw/             # 원본 데이터
│   └── processed/       # 전처리 데이터
├── notebooks/
│   ├── member1/         # 데이터 분석
│   ├── member2/         # 모델 개발
│   └── member3/         # 파생변수 실험
├── src/
│   ├── preprocess.py    # 데이터 전처리
│   ├── features.py      # 파생변수 생성
│   ├── evaluate.py      # 모델 성능 평가
│   ├── ensemble.py      # 앙상블
│   └── predict.py       # 최종 예측
├── models/
│   └── final/           # 최종 모델 (Git 제외)
├── docs/
│   ├── experiments.csv  # 실험 기록
│   └── data_dictionary.md
├── requirements.txt
└── README.md

## 5. 데이터 관리

GitHub:
- Python 코드
- Jupyter Notebook
- 문서
- 실험 기록

Google Drive:
- 원본 데이터
- 전처리된 대용량 데이터
- 학습된 모델
- 대용량 결과물

주의사항:
- 대회 데이터는 허용된 범위에서만 공유합니다.
- 데이터 및 모델 가중치는 GitHub에 업로드하지 않습니다.
- API Key 및 개인정보를 업로드하지 않습니다.
- Notebook을 업로드하기 전에 민감한 출력이 있는지 확인합니다.

Google Drive 공유 경로:
대회 시작 후 입력

## 6. GitHub 협업 규칙

각 팀원은 개인 브랜치에서 작업합니다.

브랜치 예시:
- feature/eda
- feature/model
- feature/engineering

처음 저장소 복제:

```bash
git clone https://github.com/mingyo07/Datathon-2026.git
cd Datathon-2026
```

최신 main에서 브랜치 생성:

```bash
git switch main
git pull origin main
git switch -c feature/eda
```

작업 내용 저장:

```bash
git status
git add .
git commit -m "feat: 작업 내용"
git push -u origin feature/eda
```

최초 Push 이후에는 git push만 사용해도 됩니다.

작업 완료 시:
1. GitHub에서 Pull Request 생성
2. 다른 팀원이 코드 확인
3. main에 Merge
4. 팀원들은 최신 main을 Pull

main 브랜치에 직접 Push하지 않습니다.

## 7. Google Colab 작업 규칙

1. Google Drive에서 데이터를 불러옵니다.
2. 개인 Notebook에서 분석 또는 모델 학습을 진행합니다.
3. 중요한 결과물은 Drive에 저장합니다.
4. 변경한 Notebook을 GitHub 개인 브랜치에 Push합니다.
5. 실행 결과와 변경 내용을 팀원들에게 공유합니다.

Colab에서 수정한 내용은 GitHub에 자동 반영되지 않습니다.

## 8. 실험 규칙

- 모든 모델은 동일한 Validation 분리 기준으로 평가합니다.
- 데이터 누수가 없는지 확인합니다.
- 대회에서 지정한 평가 지표를 우선 사용합니다.
- 모델 또는 파생변수를 변경했다면 결과를 기록합니다.
- 최종 Test 데이터는 모델 선택에 사용하지 않습니다.

실험 기록 파일:
docs/experiments.csv

## 9. 최종 제출 체크

- [ ] 제출 형식 확인
- [ ] 예측 결과 행 개수 확인
- [ ] 예측 파일에 결측치가 없는지 확인
- [ ] 최종 모델 및 전처리 코드 확인
- [ ] 최종 코드 실행 확인
- [ ] 발표 자료 완성
- [ ] 제출 마감 시간 확인

대회 공식 제출 규정을 최우선으로 따릅니다.
