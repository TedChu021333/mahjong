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
    choose_self_kong,
    claim_value,
    claim_widens,
    hand_value,
    declare_is_worth,
    gold_in_use,
    network_claim_is_worth,
    open_kong_is_worth,
    keep_value,
    visible_tile_counts,
)
from agent.safety import claim_is_worse, safer_discard
from agent.shanten import effective_tiles, improvement_count, next_step_outs, shanten
from game.rules import Action, ActionType, GameState, Phase, legal_actions
from game.tiles import is_honor, is_suited, suit_of, tile_name
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
                    scores = logits.cpu().numpy()
                    kept = keep_tenpai(state, player, choice, lookup, scores)
                    if kept is not None:
                        choice = kept[0]
                    wasteful = wasteful_discard(state, player, choice, lookup, scores)
                    if wasteful is not None:
                        choice = wasteful[0]
                    flush = flush_discard(state, player, choice, lookup, scores)
                    if flush is not None:
                        choice = flush[0]
                    choice = self._guarded(state, player, choice, actions, rng)
                    self.last_note = "；".join(
                        note for note in (kept and kept[1], wasteful and wasteful[1], flush and flush[1], self.last_note)
                        if note) or None
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
        if state.phase == Phase.DISCARD and not state.declared[player]:
            # 能暗槓／加槓、槓完向聽不變差就槓（規則式也是如此）：10/28 19:12 四張五條、「槓」亮著，網路卻打五筒
            kong = choose_self_kong(state, player)
            if kong is not None:
                self.last_note = "槓完向聽不變差就槓"
                return kong
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


def keep_tenpai(state: GameState, player: int, choice: Action, lookup: dict[int, Action],
                logits: np.ndarray) -> tuple[Action, str] | None:
    """沒人宣告時，網路打的牌讓聽牌退成一向聽以上，而別張打完能聽、聽的牌還有剩：改打保持聽牌的那張，回傳 (動作, 說明)。

    10/29 17:17 六萬八萬九九九萬 一二三筒 五五六七八筒 九九九條 摸四筒，打四筒聽嵌七萬（一張都沒出現），網路打六萬退成一向聽。
    台灣麻將沒有振聽、沒按聽手牌也不鎖，留著聽牌之後照樣能改聽、做大。10/28～29 沒人宣告又能聽牌的 647 手中網路拆了 5 手。
    挑聽的牌剩最多張的；一樣時不打用得上的金牌，再照網路的偏好。有人宣告時由 agent.safety 管（RETREAT_MIN_DANGER）。"""
    me = state.players[player]
    if (choice.kind not in (ActionType.DISCARD, ActionType.DECLARE) or state.phase != Phase.DISCARD
            or state.declared[player] or any(state.declared[seat] for seat in range(4) if seat != player)):
        return None
    options = [index - DISCARD for index in lookup if DISCARD <= index < DISCARD + 34]
    if choice.tile not in options:
        return None

    def after(tile: int) -> list[int]:
        hand = list(me.hand)
        hand[tile] -= 1
        return hand

    if shanten(after(choice.tile), me.open_melds) <= 0:
        return None
    visible = visible_tile_counts(state, player)
    live = {}
    for tile in options:
        hand = after(tile)
        if shanten(hand, me.open_melds) == 0:
            waits = effective_tiles(hand, me.open_melds)
            count = sum(max(0, 4 - visible[t]) for t in waits)
            if count > 0:
                live[tile] = (count, waits)
    if not live:
        return None
    pick = min(live, key=lambda tile: (-live[tile][0], tile in state.gold_tiles and gold_in_use(me.hand, tile),
                                       -float(logits[DISCARD + tile])))
    count, waits = live[pick]
    return (lookup[DISCARD + pick],
            f"保聽：打{tile_name(choice.tile)}退成一向聽 → 改打{tile_name(pick)} 聽"
            f"{''.join(tile_name(t) for t in waits)}（剩 {count} 張）")


WASTE_OUTS = 4
"""沒人宣告時，網路那張的進張比同向聽最好的少這麼多張以上就改打（聽牌時改看 WASTE_TENPAI_SHARE）。"""
WASTE_TENPAI_SHARE = 2 / 3
"""聽牌時，網路那張聽的張數不到最好的這個比例就改打。"""
SHAPE_IMPROVEMENT_MARGIN = 4
SHAPE_IMPROVEMENT_RATIO = 1.25
SHAPE_NEXT_MARGIN = 0.10
"""進張一樣時比牌型：別的牌改良牌、牌型（next_step_outs）都不比網路那張差，而且改良牌多 SHAPE_IMPROVEMENT_MARGIN 張以上、
又是 SHAPE_IMPROVEMENT_RATIO 倍以上，或牌型好 SHAPE_NEXT_MARGIN 以上就改打（10/20～10/28 記錄重放：門檻只有「多 4 張或好 5%」時
沒人宣告的出牌改 4.6%，很多是 29→33 這種小差距）。10/28 使用者：一二四打四（留一二邊張）——10/20 起這種形狀 45 次有 16 次打四，
進張全都一樣、網路照自己的偏好留么九；打一留二四，摸五變成二＋四五兩面（23:15 改良牌 16 對 8）。
孤張字牌只要改良牌多就打（不看差距）：11/1 01:32 碰九條後 一三六七七萬 一二三筒 一三條 六七八條 西，網路打六萬（拆六七七萬、
定七萬當眼），打西進張一樣 8 張、改良牌 34 對 30，使用者：眼睛定太早。10/20～11/1 兩台沒人宣告的 14312 手重放改 55 手
（打字牌的改良牌和網路那張完全一樣的 81 手照網路，可能是留安全牌）。"""


def wasteful_discard(state: GameState, player: int, choice: Action, lookup: dict[int, Action],
                     logits: np.ndarray) -> tuple[Action, str] | None:
    """沒人宣告時，網路打的牌比同向聽最好的打法進張少很多：改打進張最多的（用得上的金牌不列入），回傳 (動作, 說明)。

    使用者 10/28：網路太死板，對子好像一定要留，旁邊能組的牌反而打掉。10/20～10/28 沒人宣告的 5639 手中 71 手如此，
    例如 10/22 14:30 八八萬 二二筒 六七八筒 二三四條 四五六條 摸三筒打三筒，聽 4 張（打二筒聽 8 張）；
    10/21 23:48 二三四萬 一條 三條 五條 六六條 中中 摸三條打五條，進張 10（打三條 16）。
    只比同向聽的牌（退向聽由湊牌優先管），網路要宣告的那張也照樣比（按不按聽之後由護欄決定）。"""
    me = state.players[player]
    if (choice.kind not in (ActionType.DISCARD, ActionType.DECLARE) or state.phase != Phase.DISCARD
            or state.declared[player] or any(state.declared[seat] for seat in range(4) if seat != player)):
        return None
    options = [index - DISCARD for index in lookup if DISCARD <= index < DISCARD + 34]
    if choice.tile not in options:
        return None
    visible = visible_tile_counts(state, player)

    def evaluate(tile: int) -> tuple[int, int]:
        hand = list(me.hand)
        hand[tile] -= 1
        return shanten(hand, me.open_melds), sum(max(0, 4 - visible[t]) for t in effective_tiles(hand, me.open_melds))

    level, outs = evaluate(choice.tile)
    scored = {tile: evaluate(tile) for tile in options
              if tile != choice.tile and not (tile in state.gold_tiles and gold_in_use(me.hand, tile))}
    same = [tile for tile, value in scored.items() if value[0] == level]
    if not same:
        return None
    top = max(scored[tile][1] for tile in same)
    if (outs >= top * WASTE_TENPAI_SHARE) if level == 0 else (top - outs < WASTE_OUTS):
        return None if level == 0 else _better_shape(state, player, choice, lookup, same, scored, outs, visible)
    best = [tile for tile in same if scored[tile][1] == top]
    pick = min(best, key=lambda tile: (keep_value(me.hand, tile, state.gold_tiles), -float(logits[DISCARD + tile])))
    word = "聽" if level == 0 else "進張"
    return (lookup[DISCARD + pick],
            f"湊牌：打{tile_name(choice.tile)}{word} {outs} 張 → 改打{tile_name(pick)} {top} 張")


def _better_shape(state: GameState, player: int, choice: Action, lookup: dict[int, Action], same: list[int],
                  scored: dict[int, tuple[int, int]], outs: int, visible: list[int]) -> tuple[Action, str] | None:
    """進張不比網路那張少的牌裡，牌型、改良牌都不差、其中一項明顯較好的（見 SHAPE_IMPROVEMENT_MARGIN）。"""
    me = state.players[player]
    candidates = [tile for tile in same if scored[tile][1] >= outs]
    if not candidates:
        return None

    def shape(tile: int) -> tuple[int, int]:
        hand = list(me.hand)
        hand[tile] -= 1
        return (next_step_outs(hand, me.open_melds, visible), improvement_count(hand, me.open_melds, visible))

    base_next, base_improve = shape(choice.tile)
    better = []
    for tile in candidates:
        next_outs, improve = shape(tile)
        if next_outs >= base_next and improve >= base_improve and (
                (improve - base_improve >= SHAPE_IMPROVEMENT_MARGIN and improve >= base_improve * SHAPE_IMPROVEMENT_RATIO)
                or next_outs >= base_next * (1 + SHAPE_NEXT_MARGIN)
                or (tile >= 27 and me.hand[tile] == 1 and improve > base_improve)):
            better.append((next_outs, improve, -keep_value(me.hand, tile, state.gold_tiles), tile))
    if not better:
        return None
    next_outs, improve, _, pick = max(better)
    return (lookup[DISCARD + pick],
            f"牌型：打{tile_name(choice.tile)}改良牌 {base_improve}、牌型 {base_next} → "
            f"改打{tile_name(pick)} {improve}、{next_outs}")


FLUSH_MIN_TILES = 13
"""手牌加副露（每組算 3 張）裡同一種數牌加字牌有這麼多張以上，才往混一色／清一色做（flush_discard）。"""
FLUSH_MAX_OFF = 3
"""手上別色的數牌最多這麼多張（10/20 07:24 一三七萬 一筒 二三四五八八條 南西西東＋一組副露，字牌多就湊到 13 張，
但別色還有 4 張、離混一色很遠）。"""
FLUSH_RETREAT_MAX_LIVE = 2
"""聽的牌扣掉看得到的只剩這麼多張以下，才准拆聽牌改做清一色。"""
FLUSH_RETREAT_MIN_OUTS = 12
"""拆聽牌後清一色一向聽要有這麼多張（看不到的、同一色的）進張。10/31 00:06 那手打掉八筒後萬子進張 13 張
（二四五六七八九萬，手上都已有 1～3 張），原本想的 20 張會把它擋掉。"""
FLUSH_RETREAT_MIN_WALL = 20
"""牌牆還剩這麼多張以上才准拆聽牌做清一色。"""


def flush_suit(state: GameState, player: int) -> int | None:
    """手牌與副露往混一色／清一色靠攏時回傳那一種數牌（0 萬、1 筒、2 條），否則 None。
    副露要全是那一色或字牌；實戰讀不到自己副露的內容（只知道吃碰時用掉的手牌，見 state_parser 的佔位副露），
    知道的牌少於每組 2 張就不算，免得把別色的副露當成一色。"""
    me = state.players[player]
    meld_tiles = [tile for meld in me.melds for tile in meld.tiles]
    if len(meld_tiles) < 2 * len(me.melds):
        return None
    counts = [0, 0, 0]
    honors = 0
    for tile in range(34):
        if me.hand[tile]:
            if is_honor(tile):
                honors += me.hand[tile]
            else:
                counts[suit_of(tile)] += me.hand[tile]
    suit = max(range(3), key=lambda s: counts[s])
    if any(not is_honor(tile) and suit_of(tile) != suit for tile in meld_tiles):
        return None
    if counts[suit] + honors + 3 * len(me.melds) < FLUSH_MIN_TILES or sum(counts) - counts[suit] > FLUSH_MAX_OFF:
        return None
    return suit


def flush_discard(state: GameState, player: int, choice: Action, lookup: dict[int, Action],
                  logits: np.ndarray) -> tuple[Action, str] | None:
    """往混一色／清一色做（10/31 使用者：萬子 14 張只剩兩張筒，感覺該做清一色），回傳 (動作, 說明)：
    ① 網路打的是那一色或字牌，而打掉別色的牌向聽一樣：改打別色的牌（進張最多的；之後放槍檢查照樣管）。
    ② 網路那張打完聽牌、但聽的牌只剩 FLUSH_RETREAT_MAX_LIVE 張以下，而打掉別色的牌後手上只剩一張別色、
       同一色的進張有 FLUSH_RETREAT_MIN_OUTS 張以上（牌牆還剩 FLUSH_RETREAT_MIN_WALL 張以上）：拆聽牌做清一色（8 台）。
       10/31 00:06 二二四四四五五六六六七七七八九萬 六筒八筒，網路打九萬聽嵌七筒（上家、對家都吃了七八九筒，只剩 2 張）。
    混一色只有 4 台，不為它拆聽牌。湊牌、保聽、牌型比較都在這之前，所以改了不會被它們改回去。"""
    me = state.players[player]
    if (choice.kind not in (ActionType.DISCARD, ActionType.DECLARE) or state.phase != Phase.DISCARD
            or state.declared[player]):
        return None
    suit = flush_suit(state, player)
    if suit is None:
        return None
    options = [index - DISCARD for index in lookup if DISCARD <= index < DISCARD + 34]
    off = [tile for tile in options if is_suited(tile) and suit_of(tile) != suit]
    if choice.tile not in options or choice.tile in off or not off:
        return None
    visible = visible_tile_counts(state, player)

    def after(tile: int) -> list[int]:
        hand = list(me.hand)
        hand[tile] -= 1
        return hand

    def outs(hand: list[int], only_suit: bool = False) -> int:
        return sum(max(0, 4 - visible[t]) for t in effective_tiles(hand, me.open_melds)
                   if not only_suit or (is_suited(t) and suit_of(t) == suit))

    level = shanten(after(choice.tile), me.open_melds)
    same = [tile for tile in off if shanten(after(tile), me.open_melds) == level]
    if same:
        pick = max(same, key=lambda tile: (outs(after(tile)), float(logits[DISCARD + tile])))
        return lookup[DISCARD + pick], f"一色：打{tile_name(choice.tile)} → 改打{tile_name(pick)}（同向聽，做一色）"
    if level != 0 or state.drawable < FLUSH_RETREAT_MIN_WALL:
        return None
    live = outs(after(choice.tile))
    if live > FLUSH_RETREAT_MAX_LIVE:
        return None
    best = None
    for tile in off:
        hand = after(tile)
        others = sum(hand[t] for t in range(34) if is_honor(t) or suit_of(t) != suit)
        if others > 1 or shanten(hand, me.open_melds) != 1:
            continue
        flush_outs = outs(hand, only_suit=True)
        if flush_outs >= FLUSH_RETREAT_MIN_OUTS and (best is None or flush_outs > best[1]):
            best = (tile, flush_outs)
    if best is None:
        return None
    pick, flush_outs = best
    return (lookup[DISCARD + pick],
            f"一色：打{tile_name(choice.tile)}聽牌只剩 {live} 張 → 改打{tile_name(pick)} 做清一色（一向聽、進張 {flush_outs} 張）")


def efficient_discard(state: GameState, player: int, choice: Action, lookup: dict[int, Action],
                      logits: np.ndarray, turns: int) -> Action:
    """前 turns 手、沒有對手宣告聽牌時湊牌優先：在同向聽的牌中依序比 有效進張 → 牌型（往後多看一步，
    next_step_outs）→ 改良牌 都最多的，最後才比最孤立（孤張字牌 → 孤張么九；用得上的金牌算值得留，rule_agent.keep_value）。
    牌型排在改良牌前面（10/28 使用者）：一萬五六萬九九萬 二二四五七八筒 一二二條（碰白），打一萬、五筒都是 30 張，
    改良牌五筒多（26 對 18）就拆了四五筒，牌型一萬較好（724 對 696）。

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
            or any(state.declared[seat] for seat in range(4) if seat != player)):
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
    for measure in (next_step_outs, improvement_count):  # 平手時才算（每張約 20～50 ms）
        if len(best) > 1:
            values = {tile: measure(after(tile), me.open_melds, visible) for tile in best}
            best = [tile for tile in best if values[tile] == max(values.values())]
    pick = min(best, key=lambda tile: (keep_value(me.hand, tile, state.gold_tiles),
                                       -float(logits[DISCARD + tile])))
    if choice.tile in best and keep_value(me.hand, choice.tile, state.gold_tiles)             <= keep_value(me.hand, pick, state.gold_tiles):
        return choice
    return lookup[DISCARD + pick]
