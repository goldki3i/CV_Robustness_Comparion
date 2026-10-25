# Robustness of Pretrained Vision Models under Image Corruptions

사전학습된 **DINOv2, CLIP, ResNet-50**의 이미지 손상 강건성을 CIFAR-10에서 비교하는 컴퓨터 비전 포트폴리오 프로젝트입니다.

> **실험 상태:** 코드와 실행 파이프라인을 제공하는 프로젝트 템플릿입니다. 이 저장소의 결과표/그래프는 실제 실행 전에는 생성되지 않으며, README에는 측정하지 않은 성능 수치를 기재하지 않습니다.

## 연구 질문

1. 노이즈, 블러, 밝기 변화가 커질 때 각 모델의 분류 정확도는 어떻게 변하는가?
2. 원본 대비 정확도 감소량이 모델별·손상 유형별로 어떻게 다른가?
3. 같은 종류의 선형 분류기를 사용했을 때, 각 사전학습 모델의 특징 표현은 CIFAR-10 클래스를 얼마나 잘 구분하는가?

## 실험 설계

- **데이터:** CIFAR-10 공식 train/test split 사용
- **특징 추출기:** DINOv2 (`facebook/dinov2-base`), CLIP (`openai/clip-vit-base-patch32`), ResNet-50 (ImageNet 사전학습 가중치)
- **특징 추출:** 모든 사전학습 모델의 가중치는 고정(frozen)합니다.
- **분류기:** 모델별 특징에 같은 `StandardScaler + LogisticRegression` 파이프라인을 학습합니다. 분류기는 공식 train split의 깨끗한 이미지 특징만으로 학습합니다.
- **평가:** 같은 공식 test 이미지의 원본 및 손상본을 짝지어 평가합니다.
- **손상:** Gaussian noise, Gaussian blur, brightness change. 각 손상은 severity 1–3으로 설정합니다.
- **지표:** 조건별 accuracy, clean accuracy 대비 accuracy drop (percentage points), 손상 조건 평균 accuracy.

### 공정성 및 해석상 주의

- 공식 train/test split을 유지해 테스트 이미지가 분류기 학습에 들어가지 않도록 합니다.
- 각 모델의 권장 전처리를 사용하되, 데이터 분할·분류기 종류·분류기 하이퍼파라미터는 공통으로 유지합니다.
- 모델 구조, 사전학습 데이터, 학습 목적이 모두 다르므로 관찰된 차이를 특정 학습 방식 하나의 인과 효과로 단정하지 않습니다.
- ResNet-50은 원래 ImageNet 분류 head 대신 pooled feature를 추출합니다. 세 모델 모두 CIFAR-10용 동일 종류의 새 선형 분류기를 사용합니다.
- `--train-limit`과 `--test-limit`은 개발 및 빠른 실행을 위한 표본 수입니다. 최종 보고서에는 실제 사용한 표본 수와 설정을 명시해야 합니다.

## 설치

Python 3.10 또는 3.11 권장. CPU에서도 실행 가능하지만 모델 다운로드 및 특징 추출에 시간이 걸릴 수 있습니다. CUDA가 가능한 PyTorch 설치는 [공식 설치 안내](https://pytorch.org/get-started/locally/)를 참고하세요.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 실행

### 빠른 파이프라인 확인

```bash
python -m src.run_experiment --models resnet50 --train-limit 1000 --test-limit 200 --severities 1 --batch-size 32 --output-dir results/smoke
```

### 전체 모델·손상 조건 실험

```bash
python -m src.run_experiment --models dinov2 clip resnet50 --train-limit 10000 --test-limit 1000 --severities 1 2 3 --batch-size 32 --output-dir results/run_001
```

처음에는 `--test-limit 1000`으로 실행 시간을 확인하고, 자원이 충분하면 공식 test set 전체(10,000장) 또는 최소 5,000장으로 확대하세요. CIFAR-10 공식 test set에서 표본을 뽑을 때는 고정 seed로 클래스 비율을 유지합니다.

### 결과 파일

실행이 끝나면 `--output-dir`에 다음이 생성됩니다.

- `metrics.csv`: 모델·조건별 정확도, clean 대비 정확도 감소량
- `summary.csv`: 모델별 clean accuracy 및 평균 corrupted accuracy 요약
- `accuracy_by_severity_noise.png`, `accuracy_by_severity_blur.png`, `accuracy_by_severity_brightness.png`: 손상 유형별 강도-정확도 그래프
- `accuracy_drop.png`: clean 대비 정확도 감소량 그래프
- `run_config.json`: 사용한 인자와 실험 설정
- `docs/RESULTS_TEMPLATE.md`: 실측 결과를 정리하는 보고서 템플릿
- `features/`: 특징 캐시(재실행 시 불필요한 추출을 줄임)

## 폴더 구조

```text
cv-robustness-portfolio/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── corruptions.py
│   ├── datasets.py
│   ├── models.py
│   ├── evaluation.py
│   └── run_experiment.py
├── tests/
│   └── test_corruptions.py
└── results/                 # 실제 실험 후 생성되는 결과물
```

## 구현 포인트

1. **짝지어진 평가:** 동일한 테스트 이미지의 clean/corrupted 버전을 비교합니다.
2. **모델별 전처리:** DINOv2와 CLIP은 해당 Hugging Face image processor, ResNet-50은 ImageNet 가중치에 포함된 전처리를 사용합니다.
3. **특징 캐싱:** 추출한 특징을 디스크에 저장해 분류기/평가 분석을 반복할 때 추론을 재사용합니다.
4. **재현성:** 데이터 표본 선택과 손상 생성에 seed를 사용하고 실행 설정을 JSON으로 저장합니다.
5. **성능 분석:** accuracy drop은 `clean accuracy - corrupted accuracy`로 계산하며 단위는 percentage points(pp)입니다.

## 한계 및 다음 단계

- 현재는 세 가지 합성 손상만 다루며, 실제 카메라·압축·도메인 변화 전체를 대표하지 않습니다.
- ImageNet 사전학습 모델과 CIFAR-10 사이에는 해상도 및 도메인 차이가 있습니다.
- 선형 probe 결과는 사전학습 모델의 모든 활용 가능성을 나타내지 않습니다. 파인튜닝이나 모델별 기본 분류 방식과는 다른 질문을 평가합니다.
- 후속 작업: 더 많은 손상 유형, severity별 confidence/ECE, 특징 벡터 cosine similarity, 여러 seed 반복 및 평균±표준편차를 추가할 수 있습니다.

## 참고 자료

- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- DINOv2: https://github.com/facebookresearch/dinov2
- CLIP: https://github.com/openai/CLIP
- ResNet: He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016), https://arxiv.org/abs/1512.03385
- scikit-learn Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

## 테스트

```bash
pytest -q
```

GitHub Actions에서도 단위 테스트를 실행하도록 설정했습니다. 단위 테스트는 손상 함수와 표본 선택 로직을 확인하며, 사전학습 모델의 전체 실험 결과를 대신하지 않습니다.

## 결과 보고 원칙

실험을 실제로 실행한 뒤 `metrics.csv`와 생성된 그래프를 확인하고, **실측 결과만** 포트폴리오에 추가하세요. 실행 전에는 어떤 모델이 가장 강건하다고 단정하지 않습니다.
