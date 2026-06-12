# %% [markdown]
# # DQN 실험 실행 노트북
#
# 이 파일은 Jupyter / VS Code Notebook에서 전체 실험을 순서대로 실행하기 위해 만든 노트북입니다.
#
# 실행 순서:
#
# 1. 프로젝트 루트 탐색
# 2. 패키지와 CUDA 사용 가능 여부 확인
# 3. 테스트 실행
# 4. baseline 실행 또는 기존 결과 재사용
# 5. DQN hyperparameter sweep 실행
# 6. best config 선택 및 final evaluation 실행
# 7. baseline-DQN 비교 CSV 생성
# 8. 결과 zip 생성 및 필수 파일 확인

# %%
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import time


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "src" / "study_scheduler_env.py").exists():
            return candidate
    raise RuntimeError("프로젝트 루트를 찾지 못했습니다. 프로젝트 폴더 안에서 이 노트북을 실행하세요.")


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print("PROJECT_ROOT =", PROJECT_ROOT)

# %%
AUTO_INSTALL_PACKAGES = ["numpy", "pandas", "matplotlib", "pytest"]
missing_auto = [package for package in AUTO_INSTALL_PACKAGES if importlib.util.find_spec(package) is None]
if missing_auto:
    print("Installing missing non-torch packages:", missing_auto)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *missing_auto],
        cwd=PROJECT_ROOT,
    )

if importlib.util.find_spec("torch") is None:
    raise RuntimeError(
        "현재 Jupyter kernel에 PyTorch가 설치되어 있지 않습니다. "
        "GPU 환경에 맞는 PyTorch를 설치한 뒤 다시 실행하세요. "
        "이미 맞게 설치된 CUDA/PyTorch 환경이 바뀌지 않도록 torch는 자동 설치하지 않았습니다."
    )

import pandas as pd
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "현재 Jupyter kernel에서 CUDA를 사용할 수 없습니다. "
        "GPU를 사용할 수 있는 kernel을 선택한 뒤 다시 실행하세요."
    )

DEVICE = "cuda"
print("torch =", torch.__version__)
print("device =", DEVICE)
print("gpu =", torch.cuda.get_device_name(0))

from experiments.sweep_dqn import (
    EXPERIMENT_VERSION,
    build_grid,
    create_comparison_summary,
    make_results_zip,
    run_final_evaluation,
    run_sweep,
    write_run_manifest,
)

# %% [markdown]
# ## 실험 설정
#
# 아래 값으로 최종 실험을 실행했습니다.

# %%
HORIZON = 365
BASELINE_SEEDS = 5

TUNING_TRAIN_SEEDS = [0, 1, 2]
TUNING_EPISODES = 100
TUNING_EVAL_EPISODES = 3

FINAL_TRAIN_SEEDS = [100, 101, 102, 103, 104]
FINAL_EPISODES = 150
FINAL_EVAL_EPISODES = 5

BATCH_SIZE = 64
BUFFER_CAPACITY = 20_000
TARGET_UPDATE_STEPS = 250
EPSILON_START = 1.0

BASELINE_OUTDIR = PROJECT_ROOT / "results" / f"final_baselines_{EXPERIMENT_VERSION}"
RESULTS_ROOT = PROJECT_ROOT / "results" / f"final_dqn_sweep_{EXPERIMENT_VERSION}"
SWEEP_OUTDIR = RESULTS_ROOT / "tuning"
FINAL_OUTDIR = RESULTS_ROOT / "final"
COMPARISON_OUTDIR = RESULTS_ROOT / "comparison"

configs = build_grid()
manifest = {
    "experiment_version": EXPERIMENT_VERSION,
    "selection_metric": "episode_return_mean",
    "reward_fix": "coverage_gap_penalty를 추가해 학습을 전혀 하지 않는 정책에도 penalty가 들어가도록 수정했습니다.",
    "horizon": HORIZON,
    "baseline": {"seeds": BASELINE_SEEDS},
    "tuning": {
        "configs": len(configs),
        "train_seeds": TUNING_TRAIN_SEEDS,
        "episodes": TUNING_EPISODES,
        "eval_episodes": TUNING_EVAL_EPISODES,
    },
    "final": {
        "train_seeds": FINAL_TRAIN_SEEDS,
        "episodes": FINAL_EPISODES,
        "eval_episodes": FINAL_EVAL_EPISODES,
    },
    "batch_size": BATCH_SIZE,
    "buffer_capacity": BUFFER_CAPACITY,
    "target_update_steps": TARGET_UPDATE_STEPS,
    "epsilon_start": EPSILON_START,
    "device": DEVICE,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}
write_run_manifest(RESULTS_ROOT / "run_manifest.json", manifest)
print(json.dumps(manifest, indent=2))

# %% [markdown]
# ## 1. 업로드한 코드 테스트

# %%
start = time.time()
test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q"],
    cwd=PROJECT_ROOT,
    text=True,
    capture_output=True,
)
print(test_result.stdout)
if test_result.returncode != 0:
    print(test_result.stderr)
    raise RuntimeError("pytest가 실패했습니다. 프로젝트 파일을 확인한 뒤 다시 실행하세요.")
print("test minutes =", round((time.time() - start) / 60, 2))

# %% [markdown]
# ## 2. Baseline 실행
#
# 이미 완료된 aggregate CSV가 있으면 다시 만들지 않고 재사용합니다.

# %%
baseline_csv = BASELINE_OUTDIR / "baseline_metrics_aggregated.csv"
if baseline_csv.exists():
    print("baseline already exists:", baseline_csv)
else:
    start = time.time()
    subprocess.run(
        [
            sys.executable,
            "experiments/run_baselines.py",
            "--episodes",
            str(BASELINE_SEEDS),
            "--horizon",
            str(HORIZON),
            "--outdir",
            str(BASELINE_OUTDIR),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("baseline minutes =", round((time.time() - start) / 60, 2))

baseline_df = pd.read_csv(baseline_csv)
baseline_df

# %% [markdown]
# ## 3. DQN Hyperparameter Sweep
#
# 중간에 끊겨도 다시 Run All 하면 완료된 config/seed는 건너뛰고 이어서 진행합니다.

# %%
start = time.time()
sweep_results, best_config = run_sweep(
    outdir=SWEEP_OUTDIR,
    configs=configs,
    train_seeds=TUNING_TRAIN_SEEDS,
    episodes=TUNING_EPISODES,
    horizon=HORIZON,
    eval_episodes=TUNING_EVAL_EPISODES,
    batch_size=BATCH_SIZE,
    buffer_capacity=BUFFER_CAPACITY,
    target_update_steps=TARGET_UPDATE_STEPS,
    epsilon_start=EPSILON_START,
    device=DEVICE,
    save_models=False,
)
print("sweep hours =", round((time.time() - start) / 3600, 2))
print("best_config =", json.dumps(best_config, indent=2))
pd.read_csv(SWEEP_OUTDIR / "sweep_grouped.csv").head(10)

# %% [markdown]
# ## 4. Final Evaluation
#
# tuning seed와 final seed는 분리되어 있습니다.

# %%
start = time.time()
final_results = run_final_evaluation(
    outdir=FINAL_OUTDIR,
    best_config=best_config,
    train_seeds=FINAL_TRAIN_SEEDS,
    episodes=FINAL_EPISODES,
    horizon=HORIZON,
    eval_episodes=FINAL_EVAL_EPISODES,
    batch_size=BATCH_SIZE,
    buffer_capacity=BUFFER_CAPACITY,
    target_update_steps=TARGET_UPDATE_STEPS,
    epsilon_start=EPSILON_START,
    device=DEVICE,
)
print("final evaluation hours =", round((time.time() - start) / 3600, 2))
final_results

# %% [markdown]
# ## 5. Baseline과 DQN 비교

# %%
comparison = create_comparison_summary(
    baseline_csv=baseline_csv,
    final_dqn_results_csv=FINAL_OUTDIR / "final_dqn_results.csv",
    outdir=COMPARISON_OUTDIR,
)
comparison

# %% [markdown]
# ## 6. 결과 압축 및 필수 파일 확인

# %%
zip_path = make_results_zip(
    RESULTS_ROOT,
    PROJECT_ROOT / "results" / f"final_dqn_sweep_{EXPERIMENT_VERSION}.zip",
)
required_files = [
    baseline_csv,
    RESULTS_ROOT / "run_manifest.json",
    SWEEP_OUTDIR / "sweep_results.csv",
    SWEEP_OUTDIR / "sweep_grouped.csv",
    SWEEP_OUTDIR / "best_config.json",
    FINAL_OUTDIR / "final_dqn_results.csv",
    FINAL_OUTDIR / "final_dqn_summary.csv",
    COMPARISON_OUTDIR / "baseline_vs_dqn_summary.csv",
    zip_path,
]
missing_outputs = [path for path in required_files if not path.exists()]
if missing_outputs:
    print("생성되지 않은 필수 파일:")
    for path in missing_outputs:
        print("-", path)
    raise RuntimeError("실험은 끝났지만 필요한 결과 파일이 일부 없습니다.")

print("완료 후 가져올 파일:")
print("-", zip_path)
print("-", baseline_csv)
print("\n직접 확인한 결과 파일:")
for path in required_files[1:-1]:
    print("-", path)
