# Jupyter 실험 실행 기록

## 1. 프로젝트 폴더 업로드

GPU를 사용할 수 있는 Jupyter 환경에 아래 프로젝트 폴더를 올렸습니다.

```text
rl_study_scheduler_project/
```

업로드 zip을 직접 만들 때는 아래 폴더는 제외했습니다.

```text
.venv/
.pytest_cache/
.mplconfig/
__pycache__/
results/
```

## 2. Python 환경 준비

Jupyter kernel에서 CUDA를 사용할 수 있는 PyTorch가 잡히는지 먼저 확인했습니다.

주의한 점은 다음과 같습니다.

- 서버에 CUDA 가능한 PyTorch가 이미 있으면 `torch`를 다시 설치하지 않았습니다.
- CUDA가 잡히지 않으면 실험을 바로 중단하도록 했습니다.
- `numpy`, `pandas`, `matplotlib`, `pytest`가 없으면 notebook에서 설치하도록 했습니다.
- `torch` 설치가 필요한 경우에는 사용하는 서버의 PyTorch 설치 안내를 따르도록 했습니다.

## 3. Notebook 실행

Jupyter에서 아래 파일을 열고 전체 실행했습니다.

```text
notebooks/dqn_sweep_notebook.ipynb
```

중간에 서버 세션이 끊길 수 있어서, 이미 완료된 tuning/final run은 건너뛰고 이어서 실행되도록 구성했습니다.

## 4. 완료 후 확인한 파일

실험이 끝난 뒤 아래 결과 파일이 생성되는지 확인했습니다.

```text
results/dqn_sweep/comparison/baseline_vs_dqn_summary.csv
results/dqn_sweep/tuning/sweep_grouped.csv
results/dqn_sweep/tuning/best_config.json
results/dqn_sweep/final/final_dqn_results.csv
results/dqn_sweep/final/final_dqn_summary.csv
```

## 5. 확인한 내용

- 코드가 업로드된 환경에서 `pytest`를 통과하는지 확인했습니다.
- baseline 결과 파일이 생성되는지 확인했습니다.
- DQN sweep 결과와 best config 파일이 생성되는지 확인했습니다.
- final evaluation 결과와 baseline-DQN 비교 CSV가 생성되는지 확인했습니다.
