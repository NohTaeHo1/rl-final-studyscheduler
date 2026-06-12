# Study Scheduler RL

이 프로젝트에서는 장기 학습 스케줄링 문제를 강화학습 환경으로 만들고,
여러 baseline 정책과 DQN 정책을 비교했습니다. 하루 학습 시간을 여러 과목의
새 학습과 복습 행동에 배분하는 방식으로 환경을 구성했으며, 최종 보존 지식량,
밀린 학습량, coverage ratio, episode return 등을 함께 확인했습니다.

## 프로젝트 구조

- `src/`: 학습 스케줄링 환경, baseline 정책, DQN agent 코드가 들어 있습니다.
- `experiments/`: baseline 실행, DQN 학습, DQN sweep 실행 코드가 들어 있습니다.
- `tests/`: 환경, baseline, DQN, sweep 기본 동작을 확인하는 테스트가 들어 있습니다.
- `notebooks/`: Jupyter에서 실험을 순서대로 실행하고 결과를 확인한 노트북이 들어 있습니다.
- `models/`: 제출용으로 저장한 학습된 DQN 모델 파일이 들어 있습니다.

## 환경 설정

실험 환경은 다음과 같이 구성했습니다.

- 과목 수: 3개
- 시뮬레이션 기간: 365일
- 하루 학습 시간: 60분
- 하루 학습 블록: 15분씩 4개
- 행동: 각 블록을 새 학습 또는 복습에 배정
- 주요 지표: 최종 보존 지식량, 최종 backlog, coverage ratio, retained-knowledge AUC, episode return

## 실행 방법

새 가상환경을 만든 뒤 필요한 패키지를 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

기본 테스트는 다음 명령어로 실행할 수 있습니다.

```bash
python -m pytest
```

Baseline 정책 실험은 다음과 같이 실행합니다.

```bash
python experiments/run_baselines.py --output-dir results/local_baselines
```

DQN 정책 하나를 학습하고 평가할 때는 다음 명령어를 사용합니다.

```bash
python experiments/train_dqn.py --output-dir results/local_dqn
```

작은 sweep 실험은 다음과 같이 실행할 수 있습니다.

```bash
python experiments/sweep_dqn.py --output-dir results/local_sweep
```

## 최종 실험 결과

학습된 DQN 모델 파일은 다음 위치에 저장했습니다.

- [seed 100 모델 다운로드](models/dqn_policy_seed_100.pt)
- [seed 101 모델 다운로드](models/dqn_policy_seed_101.pt)
- [seed 102 모델 다운로드](models/dqn_policy_seed_102.pt)
- [seed 103 모델 다운로드](models/dqn_policy_seed_103.pt)
- [seed 104 모델 다운로드](models/dqn_policy_seed_104.pt)

## 결과 해석

DQN은 episode return 기준으로 가장 좋은 값을 보였습니다. 특히 backlog를 낮게
유지하면서 scalar reward를 크게 만드는 방향으로 학습되었습니다.

하지만 여러 seed에서 retained knowledge와 coverage ratio가 낮게 나오는 문제가
있었습니다. 이는 DQN이 새 학습을 충분히 늘리기보다 backlog를 줄이는 쪽으로
행동하면서 생긴 한계로 해석했습니다. 반대로 `greedy_immediate_gain` baseline은
최종 보존 지식량과 retained-knowledge AUC 관점에서 더 좋은 결과를 보였습니다.

따라서 보고서에서는 DQN이 전체적으로 가장 좋은 스케줄러라고 주장하지 않았습니다.
대신 reward 설계와 coverage 문제 때문에 episode return과 실제 학습 성과 사이에
차이가 생길 수 있다는 점을 중심으로 해석했습니다.

## 코드 확인 및 테스트

제출 전에 새 가상환경에서 `python -m pytest`를 실행해 기본 동작을 확인했습니다.
전체 테스트는 통과했습니다.

테스트는 크게 네 부분으로 나누어 작성했습니다. 환경 테스트에서는 observation
shape, action space, 새 학습 item 생성, review 업데이트, episode 종료 조건을
확인했습니다. Baseline 테스트에서는 각 baseline policy가 유효한 action을 내고
짧은 episode를 끝까지 실행하는지 확인했습니다. DQN 테스트에서는 replay buffer,
action 선택, 짧은 학습 및 평가 함수가 정상적으로 실행되는지 확인했습니다. Sweep
테스트에서는 작은 설정에서 sweep 결과, final evaluation 결과, baseline-DQN 비교
파일이 생성되는지 확인했습니다.
