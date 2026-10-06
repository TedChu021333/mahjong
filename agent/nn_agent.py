"""學習型 AI：用訓練好的網路（training.model）做決定，介面與其他 AI 相同
（policy(state, player, rng) -> Action），可以直接放進模擬器、arena 與提示模式。

摸牌與換三張不交給網路，沿用規則式 AI。需要 PyTorch（.venv-train）。
"""
from __future__ import annotations

from random import Random

import numpy as np
import torch

from agent.rule_agent import choose_rule_action_for_player
from game.rules import Action, GameState, legal_actions
from training.actions import legal_mask
from training.encoding import encode
from training.model import MahjongNet, load, masked_logits


class NetworkPolicy:
    def __init__(self, model: MahjongNet, device: str = "cpu", temperature: float = 0.0) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature
        """0 = 一律選分數最高的動作；>0 依 softmax(分數 / temperature) 抽樣（自我對打探索用）。"""

    @classmethod
    def from_file(cls, path, device: str = "cpu", temperature: float = 0.0) -> "NetworkPolicy":
        return cls(load(path, device), device, temperature)

    def __call__(self, state: GameState, player: int, rng: Random) -> Action | None:
        actions = legal_actions(state, player)
        if not actions:
            return None
        mask, lookup = legal_mask(actions)
        if not lookup:
            return choose_rule_action_for_player(state, player, rng)  # 摸牌、換三張
        if len(lookup) == 1:
            return next(iter(lookup.values()))
        obs = torch.from_numpy(encode(state, player)[None]).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(obs)
        logits = masked_logits(logits, torch.from_numpy(mask[None]).to(self.device))[0]
        if self.temperature <= 0:
            return lookup[int(logits.argmax())]
        probs = torch.softmax(logits / self.temperature, 0).cpu().numpy().astype(np.float64)
        indices = np.nonzero(mask)[0]
        weights = probs[indices] / probs[indices].sum()
        return lookup[int(rng.choices(list(indices), weights=list(weights))[0])]
