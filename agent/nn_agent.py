"""學習型 AI：用訓練好的網路（training.model）做決定，介面與其他 AI 相同
（policy(state, player, rng) -> Action），可以直接放進模擬器、arena 與提示模式。

摸牌與換三張不交給網路，沿用規則式 AI。需要 PyTorch（.venv-train）。
"""
from __future__ import annotations

from random import Random

import numpy as np
import torch

from agent.rule_agent import (
    choose_rule_action_for_player,
    declare_is_worth,
    gold_in_use,
    open_kong_is_worth,
    keep_value,
    visible_tile_counts,
)
from agent.safety import claim_is_worse, safer_discard
from agent.shanten import effective_tiles, shanten
from game.rules import Action, ActionType, GameState, Phase, legal_actions
from training.actions import DISCARD, legal_mask
from training.encoding import encode
from training.model import MahjongNet, load, masked_logits


class NetworkPolicy:
    def __init__(self, model: MahjongNet, device: str = "cpu", temperature: float = 0.0,
                 efficiency_turns: int = 0, guard: bool = False) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature
        """0 = 一律選分數最高的動作；>0 依 softmax(分數 / temperature) 抽樣（自我對打探索用）。"""
        self.efficiency_turns = efficiency_turns
        """前幾手、沒人宣告聽牌時湊牌優先（見 efficient_discard）；0 = 不用。"""
        self.guard = guard
        """實戰用（提示模式預設開啟）：按聽照規則式（地聽或聽牌型態夠好才按，declare_is_worth），有對手宣告時
        用期望值檢查打牌與吃碰（agent.safety）。10/17 實戰 45 局按聽 11 次、放槍 10 次（6 次放給已宣告的人），
        使用者 10/5 也說過太容易按聽。自我對打訓練不開。"""
        self.last_note: str | None = None
        """上一個決定被 guard 改掉的原因（記錄用）。"""

    @classmethod
    def from_file(cls, path, device: str = "cpu", temperature: float = 0.0,
                  efficiency_turns: int = 0, guard: bool = False) -> "NetworkPolicy":
        return cls(load(path, device), device, temperature, efficiency_turns, guard)

    def __call__(self, state: GameState, player: int, rng: Random) -> Action | None:
        self.last_note = None
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
            if self.guard:
                try:
                    choice = self._guarded(state, player, choice, actions, rng)
                except ValueError:  # 局面不合法（例如辨識出的張數不對）：照網路
                    self.last_note = None
            return choice
        probs = torch.softmax(logits / self.temperature, 0).cpu().numpy().astype(np.float64)
        indices = np.nonzero(mask)[0]
        weights = probs[indices] / probs[indices].sum()
        return lookup[int(rng.choices(list(indices), weights=list(weights))[0])]

    def _guarded(self, state: GameState, player: int, choice: Action, actions: list[Action],
                 rng: Random) -> Action:
        notes = []
        if choice.kind == ActionType.KONG and state.phase == Phase.RESPONSE                 and not open_kong_is_worth(state.players[player].melds):
            self.last_note = "門清不明槓"
            return next(action for action in actions if action.kind == ActionType.PASS)
        if choice.kind in (ActionType.CHOW, ActionType.PUNG):
            note = claim_is_worse(state, player, choice, rng)
            if note is not None:
                self.last_note = note
                return next(action for action in actions if action.kind == ActionType.PASS)
            return choice
        if choice.kind not in (ActionType.DISCARD, ActionType.DECLARE):
            return choice
        tile, note = safer_discard(state, player, choice.tile, rng)
        if note is not None:
            notes.append(note)
        # 按不按聽照規則（rule_agent.declare_is_worth：地聽、或聽牌型態夠好），不看網路想不想按
        declare = Action(ActionType.DECLARE, tile=tile)
        if declare in actions and declare_is_worth(state, player, tile):
            result = declare
            if choice.kind != ActionType.DECLARE:
                notes.append("按聽（聽牌型態夠好）")
        else:
            if choice.kind == ActionType.DECLARE:
                notes.append("不按聽（聽牌型態還能改善或剩的牌太少）")
            result = Action(ActionType.DISCARD, tile=tile)
        self.last_note = "；".join(notes) or None
        return result


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
