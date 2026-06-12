from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
from torch import nn


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.storage: Deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.storage.append(
            Transition(
                obs=obs.astype(np.float32, copy=True),
                action=int(action),
                reward=float(reward),
                next_obs=next_obs.astype(np.float32, copy=True),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.storage):
            raise ValueError("batch_size is larger than the current replay buffer.")
        return self.rng.sample(list(self.storage), batch_size)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 20_000,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        torch.manual_seed(seed)
        random.seed(seed)

        self.q_net = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_capacity, seed=seed)

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        obs = torch.as_tensor(
            np.asarray([transition.obs for transition in batch], dtype=np.float32),
            device=self.device,
        )
        actions = torch.as_tensor(
            [transition.action for transition in batch],
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [transition.reward for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_obs = torch.as_tensor(
            np.asarray([transition.next_obs for transition in batch], dtype=np.float32),
            device=self.device,
        )
        dones = torch.as_tensor(
            [transition.done for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )

        q_values = self.q_net(obs).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_obs).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())
