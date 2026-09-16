# TriFusion-FP-Kernel

**TriFusion-FP 오디오 모델을 위한 NVIDIA L4 추론 커널 라이브러리**입니다.

## 현재 구현

- 큰 행렬 곱·합성곱: PyTorch가 호출하는 cuBLAS/cuDNN
- Mamba-3, normalization, gate, mask, 중간 버퍼 연산: Triton GPU 커널
- 일부 수학 함수: Triton 안에서 실행되는 upstream PTX
- 반복 실행: CUDA Graph
- 입력 디코딩·배치 구성: Python/NumPy/SciPy/SoundFile

전부 C++로 옮긴 라이브러리는 아닙니다. 핵심 계산은 GPU에서 수행하고,
Python 호출 비용은 CUDA Graph로 줄였습니다. 이 버전은 **추론 전용**이며
학습용 backward나 C++ ABI를 제공하지 않습니다.

## 설치와 사용

실측 환경은 Linux / NVIDIA L4(SM89) / Python 3.11.15 /
PyTorch 2.7.1+cu128 / Triton 3.3.1입니다.

```bash
git clone https://github.com/tlstngud/TriFusion-FP-Kernel.git
cd TriFusion-FP-Kernel
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install '.[l4]'
trifusion-l4 info
trifusion-l4 predict --weights /path/to/model.pt --root /path/to/work
```

기본 패키지(`pip install .`)는 PyTorch를 설치하지 않습니다. GPU 추론은
검증한 버전을 지정한 `l4` extra로 설치합니다. 모델 가중치와 오디오 데이터는
포함하지 않으므로 호환되는 체크포인트를 직접 준비해야 합니다.

```python
from trifusion_l4 import configure_runtime, load_student, InferenceEngine

configure_runtime()
model = load_student("/path/to/model.pt")
engine = InferenceEngine(model, graphs=True, max_graphs=1)
probabilities = engine(audio, lengths, channels).float().sigmoid()
engine.clear()
```

[체크포인트·텐서 입력 형식](docs/api.md), [벤치마크와 검증 범위](docs/benchmarks.md),
[전체 설명](README.md)을 확인할 수 있습니다. GitHub에서 설치할 수 있으며
PyPI에는 아직 배포하지 않았습니다.

기존 L4 실측에서 동일 GPU 기준 구현 대비 약 1.9~4.8배 향상했습니다
(입력/배치별 차이). 이 수치는 공개 패키지로 정리하기 전의 비공개 가중치와
오디오에 대한 측정입니다. 가중치·데이터가 없어 해당 전체 모델 측정값을
이 저장소만으로 그대로 재현할 수는 없습니다. 합성 입력 커널 테스트와
자신의 가중치를 사용하는 벤치마크 명령을 제공합니다.

라이선스: **Apache-2.0**. Mamba-3 원본과 파생 코드의 출처·저작권은
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 기록했습니다.
