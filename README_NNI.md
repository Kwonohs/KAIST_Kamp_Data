# LSTM Autoencoder 이상 탐지 - NNI 최적화

이 프로젝트는 LSTM Autoencoder를 사용한 이상 탐지 모델의 하이퍼파라미터를 NNI(Neural Network Intelligence)로 최적화합니다.

## 파일 구조

```
my-project/
├── config.yml              # NNI 설정 파일
├── trial.py                # NNI trial 함수 (메인 모델 코드)
├── run_experiment.py       # 실험 실행 스크립트
├── requirements.txt        # 필요한 패키지 목록
├── README_NNI.md          # 이 파일
├── dataset/
│   ├── outlier_data.csv    # 이상 데이터
│   └── press_data_normal.csv # 정상 데이터
└── test.ipynb             # 원본 Jupyter 노트북
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# NNI 설치 확인
nnictl --version
```

### 2. 데이터 준비

`dataset/` 폴더에 다음 파일들이 있어야 합니다:
- `outlier_data.csv`: 이상 데이터
- `press_data_normal.csv`: 정상 데이터

### 3. 실험 실행

```bash
# 방법 1: Python 스크립트 사용 (권장)
python run_experiment.py

# 방법 2: 직접 NNI 명령어 사용
nnictl create --config config.yml
```

### 4. 실험 모니터링

실험이 실행되면 다음 URL에서 실시간 모니터링이 가능합니다:
- http://localhost:8080

## 최적화되는 하이퍼파라미터

- **LSTM 유닛 수**: 32, 64, 128, 256
- **학습률**: 0.0001 ~ 0.01 (로그 스케일)
- **배치 크기**: 32, 64, 128, 256
- **드롭아웃 비율**: 0.0 ~ 0.5
- **시퀀스 길이**: 10, 20, 30, 40
- **오프셋**: 50, 100, 150, 200

## 평가 지표

각 trial에서 다음 지표들을 계산합니다:
- **Accuracy**: 정확도
- **Precision**: 정밀도
- **Recall**: 재현율
- **F1-Score**: F1 점수
- **Threshold**: 최적 임계값

## 실험 설정

- **최대 Trial 수**: 50
- **최대 실험 시간**: 2시간
- **동시 Trial 수**: 2
- **Tuner**: TPE (Tree-structured Parzen Estimator)
- **Assessor**: Medianstop

## 결과 해석

실험 완료 후 NNI Web UI에서:
1. **Overview**: 전체 실험 진행 상황
2. **Trials detail**: 각 trial의 상세 결과
3. **Hyper parameter**: 하이퍼파라미터별 성능 분석
4. **Trial duration**: 각 trial의 실행 시간

## 주요 개선사항

1. **모듈화**: 코드를 함수별로 분리하여 재사용성 향상
2. **하이퍼파라미터 최적화**: NNI를 통한 자동 최적화
3. **성능 지표**: 다양한 평가 지표 추가
4. **에러 처리**: 안정적인 실행을 위한 에러 처리
5. **모니터링**: 실시간 실험 진행 상황 확인

## 사용법 예시

```python
# 개별 trial 실행 (디버깅용)
python trial.py

# 전체 실험 실행
python run_experiment.py

# 실험 중단
nnictl stop

# 실험 결과 확인
nnictl experiment list
```

## 주의사항

1. GPU 메모리가 충분한지 확인하세요
2. 데이터 파일 경로가 올바른지 확인하세요
3. 실험 중에는 다른 GPU 작업을 피하세요
4. 실험 완료 후 불필요한 모델 파일을 정리하세요
