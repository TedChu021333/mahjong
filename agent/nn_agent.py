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
    claim_value,
    claim_widens,
    hand_value,
    declare_is_worth,
    network_claim_is_worth,
    honor_tai,
    open_kong_is_worth,
    keep_value,
    visible_tile_counts,
)
from agent.safety import anyone_declared, attack_discard, claim_is_worse, safer_discard
from agent.shanten import effective_tiles, improvement_count, next_step_outs, shanten
from game.rules import Action, ActionType, GameState, Phase, legal_actions
from training.actions import DISCARD, legal_mask
from training.encoding import encode
from training.model import MahjongNet, load, masked_logits


ATTACK_TURNS = 99
"""極致進攻時湊牌優先整局都用。"""
ATTACK_FOLD_SHANTEN = 2
"""極致進攻：自己還有這麼多向聽以上、又有人宣告時，暫停進攻、照一般護欄（10/22 極致進攻放給已宣告者 21 次，
好幾次是兩、三向聽根本快胡不了的手）。"""


class NetworkPolicy:
    def __init__(self, model: MahjongNet, device: str = "cpu", temperature: float = 0.0,
                 efficiency_turns: int = 0, guard: bool = False, attack: bool = False) -> None:
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
        self.attack = attack
        """極致進攻（使用者 10/22 要求，提示模式 --attack）：整局湊牌優先（有人宣告也一樣）；有人宣告時只擋放槍
        超過 safety.ATTACK_MAX_DANGER 的牌；進張變多就吃碰（向聽變好、或向聽相同進張變多），不做吃碰後的放槍檢查；
        按聽、門清不明槓照一般規則。需要 guard=True。"""
        self.last_note: str | None = None
        """上一個決定被 guard 改掉的原因（記錄用）。"""

    @classmethod
    def from_file(cls, path, device: str = "cpu", temperature: float = 0.0,
                  efficiency_turns: int = 0, guard: bool = False, attack: bool = False) -> "NetworkPolicy":
        return cls(load(path, device), device, temperature, efficiency_turns, guard, attack)

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
            if self._attacking(state, player):
                choice = efficient_discard(state, player, choice, lookup, logits.cpu().numpy(),
                                           ATTACK_TURNS, ignore_declared=True)
            elif self.efficiency_turns:
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
        attack = self._attacking(state, player)
        paused = getattr(self, "attack", False) and not attack
        if attack and state.phase == Phase.RESPONSE and choice.kind in (
                ActionType.CHOW, ActionType.PUNG, ActionType.PASS):
            return self._attack_claim(state, player, choice, actions)
        if choice.kind in (ActionType.CHOW, ActionType.PUNG)                 and not network_claim_is_worth(state, player, choice):
            self.last_note = "吃碰不會讓向聽變好"
            return next(action for action in actions if action.kind == ActionType.PASS)
        if choice.kind == ActionType.PASS and state.phase == Phase.RESPONSE:
            widen = widening_claim(state, player, actions)
            if widen is not None and claim_is_worse(state, player, widen[0], rng) is None:
                self.last_note = widen[1]
                return widen[0]
            return choice
        if choice.kind in (ActionType.CHOW, ActionType.PUNG):
            note = claim_is_worse(state, player, choice, rng)
            if note is not None:
                self.last_note = note
                return next(action for action in actions if action.kind == ActionType.PASS)
            return choice
        if choice.kind not in (ActionType.DISCARD, ActionType.DECLARE):
            return choice
        check = attack_discard if attack else safer_discard
        tile, note = check(state, player, choice.tile, rng)
        if note is not None:
            notes.append(f"極致進攻暫停（兩向聽以上又有人宣告）：{note}" if paused else note)
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


    def _attacking(self, state: GameState, player: int) -> bool:
        """這一步要不要用極致進攻：有開，而且不是「有人宣告、自己還有兩向聽以上」。"""
        if not getattr(self, "attack", False):
            return False
        me = state.players[player]
        try:
            far = shanten(me.hand, me.open_melds) >= ATTACK_FOLD_SHANTEN
        except ValueError:
            return True
        return not (far and anyone_declared(state, player))

    def _attack_claim(self, state: GameState, player: int, choice: Action, actions: list[Action]) -> Action:
        """極致進攻的吃碰：進張變多就吃碰（loose，加台的字牌碰了向聽不變差也碰），網路不吃碰時改成最好的那個。"""
        if state.last_discard is None or state.declared[player]:
            return choice
        me = state.players[player]
        before = hand_value(me.hand, me.open_melds)
        best = None
        for action in actions:
            if action.kind not in (ActionType.CHOW, ActionType.PUNG):
                continue
            after = claim_value(me.hand, me.open_melds, action, state.last_discard)
            honor = action.kind == ActionType.PUNG and honor_tai(state, player, state.last_discard)
            if (after < before or (honor and after[0] <= before[0])) and (best is None or after < best[1]):
                best = (action, after)
        passing = next(action for action in actions if action.kind == ActionType.PASS)
        if best is None:
            if choice.kind != ActionType.PASS:
                self.last_note = "極致進攻：吃碰後進張沒變多，不要"
            return passing
        if choice.kind == ActionType.PASS or claim_value(me.hand, me.open_melds, choice,
                                                         state.last_discard) > best[1]:
            word = "碰" if best[0].kind == ActionType.PUNG else "吃"
            self.last_note = (f"極致進攻：{word}了 向聽 {before[0]} → {best[1][0]}、"
                              f"進張 {-before[1]} → {-best[1][1]} 張")
            return best[0]
        return choice


def widening_claim(state: GameState, player: int, actions: list[Action]) -> tuple[Action, str] | None:
    """網路不吃碰、但有吃碰能讓一向聽的進張大增（rule_agent.claim_widens）時改成吃碰：(動作, 記錄說明)。
    10/21 23:43 一二二條 四五五條，只等三條 4 張，碰二條後 32 張，網路自己不碰。"""
    if state.last_discard is None or state.declared[player]:
        return None
    me = state.players[player]
    before = hand_value(me.hand, me.open_melds)
    best = None
    for action in actions:
        if action.kind not in (ActionType.CHOW, ActionType.PUNG):
            continue
        after = claim_value(me.hand, me.open_melds, action, state.last_discard)
        if claim_widens(before, after) and (best is None or after < best[1]):
            best = (action, after)
    if best is None:
        return None
    word = "碰" if best[0].kind == ActionType.PUNG else "吃"
    return best[0], f"{word}了進張 {-before[1]} → {-best[1][1]} 張（一向聽）"


def efficient_discard(state: GameState, player: int, choice: Action, lookup: dict[int, Action],
                      logits: np.ndarray, turns: int, ignore_declared: bool = False) -> Action:
    """前 turns 手、沒有對手宣告聽牌時湊牌優先：在同向聽的牌中依序比 有效進張 → 改良牌 → 牌型（往後多看一步，
    next_step_outs）都最多的，最後才比最孤立（孤張字牌 → 孤張么九；用得上的金牌算值得留，rule_agent.keep_value）。

    10/15 實戰：網路學到前期先打中張、留字牌與么九當後期的安全牌（規則式打孤張字牌的局面它 42% 改打
    數牌），例如 六八九萬 打六萬比打孤張東少 4 張進張。使用者認為湊牌應該優先。宣告聽牌、向聽不同、
    中後期與有人宣告時都照網路的判斷。
    10/20：原本用得上的金牌不列入候選，三四六七九筒（七九筒是金牌）進張、改良牌都一樣時打了六筒、留七九嵌張；
    使用者要牌型勝過金牌，改成金牌也比，打九筒留兩組兩面。
    10/21 21:34：五六萬 二三三四筒 一二二四五六八九條 發發 南（莊家，南沒用），網路打六萬退成 3 向聽（打南是 2 向聽）；
    原本只在「和網路那張同向聽」的牌裡挑，網路退向聽就攔不到。前期沒人宣告時沒有理由退向聽，改成從向聽最好的牌裡挑。"""
    me = state.players[player]
    if (choice.kind != ActionType.DISCARD or state.phase != Phase.DISCARD
            or max(len(me.discards), state.min_discards) >= turns
            or (not ignore_declared and any(state.declared[seat] for seat in range(4) if seat != player))):
        return choice
    visible = visible_tile_counts(state, player)

    def after(tile: int) -> list[int]:
        hand = list(me.hand)
        hand[tile] -= 1
        return hand

    def evaluate(tile: int) -> tuple[int, int]:
        hand = after(tile)
        outs = sum(max(0, 4 - visible[t]) for t in effective_tiles(hand, me.open_melds))
        return shanten(hand, me.open_melds), outs

    options = [index - DISCARD for index in lookup if DISCARD <= index < DISCARD + 34]
    scored = {tile: evaluate(tile) for tile in options}
    level = min(value[0] for value in scored.values())
    same = [tile for tile in options if scored[tile][0] == level]
    top = max(scored[tile][1] for tile in same)
    best = [tile for tile in same if scored[tile][1] == top]
    for measure in (improvement_count, next_step_outs):  # 平手時才算（每張約 20～50 ms）
        if len(best) > 1:
            values = {tile: measure(after(tile), me.open_melds, visible) for tile in best}
            best = [tile for tile in best if values[tile] == max(values.values())]
    pick = min(best, key=lambda tile: (keep_value(me.hand, tile, state.gold_tiles),
                                       -float(logits[DISCARD + tile])))
    if choice.tile in best and keep_value(me.hand, choice.tile, state.gold_tiles)             <= keep_value(me.hand, pick, state.gold_tiles):
        return choice
    return lookup[DISCARD + pick]
