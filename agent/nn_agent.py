"""學習型 AI：用訓練好的網路（training.model）做決定，介面與其他 AI 相同
（policy(state, player, rng) -> Action），可以直接放進模擬器、arena 與提示模式。

摸牌與換三張不交給網路，沿用規則式 AI。需要 PyTorch（.venv-train）。
"""
from __future__ import annotations

from random import Random

import numpy as np
import torch

from agent.rule_agent import choose_rule_action_for_player, gold_in_use, keep_value, visible_tile_counts
from agent.shanten import effective_tiles, shanten
from game.rules import Action, ActionType, GameState, Phase, legal_actions
from training.actions import DISCARD, legal_mask
from training.encoding import encode
from training.model import MahjongNet, load, masked_logits


class NetworkPolicy:
    def __init__(self, model: MahjongNet, device: str = "cpu", temperature: float = 0.0,
                 efficiency_turns: int = 0) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature
        """0 = 一律選分數最高的動作；>0 依 softmax(分數 / temperature) 抽樣（自我對打探索用）。"""
        self.efficiency_turns = efficiency_turns
        """前幾手、沒人宣告聽牌時湊牌優先（見 efficient_discard）；0 = 不用。"""

    @classmethod
    def from_file(cls, path, device: str = "cpu", temperature: float = 0.0,
                  efficiency_turns: int = 0) -> "NetworkPolicy":
        return cls(load(path, device), device, temperature, efficiency_turns)

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
            choice = lookup[int(logits.argmax())]
            if self.efficiency_turns:
                choice = efficient_discard(state, player, choice, lookup, logits.cpu().numpy(),
                                           self.efficiency_turns)
            return choice
        probs = torch.softmax(logits / self.temperature, 0).cpu().numpy().astype(np.float64)
        indices = np.nonzero(mask)[0]
        weights = probs[indices] / probs[indices].sum()
        return lookup[int(rng.choices(list(indices), weights=list(weights))[0])]


def efficient_discard(state: GameState, player: int, choice: Action, lookup: dict[int, Action],
                      logits: np.ndarray, turns: int) -> Action:
    """前 turns 手、沒有對手宣告聽牌時湊牌優先：網路要打的牌若比同向聽的其他牌少進張，改打進張最多的；
    進張也一樣時先打最孤立的牌（孤張字牌 → 孤張么九，rule_agent.keep_value）。用得上的金牌不換掉。

    10/15 實戰：網路學到前期先打中張、留字牌與么九當後期的安全牌（規則式打孤張字牌的局面它 42% 改打
    數牌），例如 六八九萬 打六萬比打孤張東少 4 張進張。使用者認為湊牌應該優先。宣告聽牌、向聽不同、
    中後期與有人宣告時都照網路的判斷。"""
    me = state.players[player]
    if (choice.kind != ActionType.DISCARD or state.phase != Phase.DISCARD
            or max(len(me.discards), state.min_discards) >= turns
            or any(state.declared[seat] for seat in range(4) if seat != player)):
        return choice
    visible = visible_tile_counts(state, player)
    gold = set(state.gold_tiles)

    def evaluate(tile: int) -> tuple[int, int]:
        after = list(me.hand)
        after[tile] -= 1
        outs = sum(max(0, 4 - visible[t]) for t in effective_tiles(after, me.open_melds))
        return shanten(after, me.open_melds), outs

    options = [index - DISCARD for index in lookup if DISCARD <= index < DISCARD + 34]
    scored = {tile: evaluate(tile) for tile in options}
    level = scored[choice.tile][0]
    same = [tile for tile in options if scored[tile][0] == level
            and (tile == choice.tile or not (tile in gold and gold_in_use(me.hand, tile)))]
    top = max(scored[tile][1] for tile in same)
    best = [tile for tile in same if scored[tile][1] == top]
    pick = min(best, key=lambda tile: (keep_value(me.hand, tile, state.gold_tiles),
                                       -float(logits[DISCARD + tile])))
    if choice.tile in best and keep_value(me.hand, choice.tile, state.gold_tiles) \
            <= keep_value(me.hand, pick, state.gold_tiles):
        return choice
    return lookup[DISCARD + pick]
