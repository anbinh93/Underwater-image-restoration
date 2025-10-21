from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EnhancementAction:
    gamma_delta: float
    clahe_delta: float
    msr_mix_delta: float
    lf_hf_alpha_delta: float


def discretize_state(state: Sequence[float], precision: int = 2) -> Tuple[float, ...]:
    return tuple(np.round(state, precision))


class QLearningAgent:
    def __init__(
        self,
        action_space: List[EnhancementAction],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(action_space), dtype=np.float32))

    def select_action(self, state: Sequence[float]) -> int:
        state_key = discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_space))
        return int(np.argmax(self.q_table[state_key]))

    def update(
        self,
        state: Sequence[float],
        action_idx: int,
        reward: float,
        next_state: Sequence[float],
    ) -> None:
        state_key = discretize_state(state)
        next_key = discretize_state(next_state)
        q_values = self.q_table[state_key]
        max_next = np.max(self.q_table[next_key])
        target = reward + self.gamma * max_next
        q_values[action_idx] = (1 - self.alpha) * q_values[action_idx] + self.alpha * target
