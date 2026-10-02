"""以向聽數與有效進張為核心的簡單規則式 AI。"""
from __future__ import annotations

from random import Random
from typing import Sequence

from .shanten import effective_tiles, shanten, standard_shanten
from game.rules import (
    MAX_SWAP,
    Action,
    ActionType,
    GameState,
    Phase,
    allowed_discards,
    forbidden_after_claim,
    legal_actions,
)
from game.tiles import is_honor, is_suited
from game.win import winning_tiles


def choose_discard(hand: list[int], n_open_melds: int = 0,
                   rng: Random | None = None,
                   visible_counts: Sequence[int] | None = None,
                   safe_tiles: Sequence[int] = (),
                   suji_tiles: Sequence[int] = (),
                   defensive: bool = False,
                   forbidden: Sequence[int] = (),
                   gold_tiles: Sequence[int] = (),
                   gold_penalty: float | None = None) -> int:
    """選一張牌打出：最低向聽優先，再取有效進張最多；forbidden 是吃碰後不能打的牌。

    金牌留在手上胡牌時每張 +1 台，打出去被胡則對方多 3 台，所以打「用得上」的金牌（有對子、
    刻子或相鄰數牌）的代價視同少了 gold_penalty 張有效進張；孤張金牌多半湊不進牌型，不加代價。
    聽牌時聽的牌有金牌（胡金牌 +3 台）則加 gold_penalty。
    防守時同樣危險程度下也先打非金牌。
    """
    if gold_penalty is None:
        gold_penalty = GOLD_DISCARD_PENALTY
    gold = set(gold_tiles)
    if sum(hand) not in (17 - 3 * n_open_melds, 16 - 3 * n_open_melds):
        raise ValueError("選擇打牌時手牌張數不正確")
    if visible_counts is not None and (
        len(visible_counts) != 34 or any(count < 0 or count > 4 for count in visible_counts)
    ):
        raise ValueError("可見牌計數必須是長度 34 且每格介於 0-4")
    candidates = []
    for tile in allowed_discards(hand, tuple(forbidden)):
        after_discard = hand.copy()
        after_discard[tile] -= 1
        next_shanten = shanten(after_discard, n_open_melds)
        candidates.append((next_shanten, tile, after_discard))
    if not candidates:
        raise ValueError("沒有可打出的牌")
    safe_set = set(safe_tiles)
    if any(tile < 0 or tile >= 34 for tile in safe_set):
        raise ValueError("安全牌編碼必須在 0-33 之間")
    suji_set = set(suji_tiles)
    if any(tile < 0 or tile >= 34 for tile in suji_set):
        raise ValueError("筋牌編碼必須在 0-33 之間")
    best = []
    for next_shanten, tile, after_discard in candidates:
        if next_shanten != min(candidate[0] for candidate in candidates):
            continue
        effective = effective_tiles(after_discard, n_open_melds)
        if visible_counts is None:
            outs = len(effective)
        else:
            outs = sum(max(0, 4 - visible_counts[tile]) for tile in effective)
        risk = 0 if tile in safe_set else 1 if tile in suji_set else 2
        is_gold = tile in gold
        value = outs - (gold_penalty if is_gold and gold_in_use(hand, tile) else 0)
        if next_shanten == 0 and gold.intersection(effective):
            value += gold_penalty  # 聽金牌：胡的那張是金牌再多 3 台
        if defensive:
            best.append(((risk, is_gold), -value, next_shanten, tile))
        else:
            best.append((-value, risk, next_shanten, tile))
    best_score = min(score[:3] for score in best)
    best_tiles = [tile for *score, tile in best if tuple(score) == best_score]
    # 平手時先打最孤立的牌（孤張字牌 → 孤張么九 → 其他），再隨機
    lowest = min(keep_value(hand, tile, gold) for tile in best_tiles)
    best_tiles = [tile for tile in best_tiles if keep_value(hand, tile, gold) == lowest]
    return (rng or Random()).choice(best_tiles)


def keep_value(hand: Sequence[int], tile: int, gold_tiles=()) -> int:
    """這張牌留著的價值，只用來打破平手：同種越多、前後兩格內的牌越多越高；
    孤張字牌最低（只能碰、湊不成順子，實戰曾留著孤張東西去拆萬子），孤張么九次之；
    用得上的金牌（gold_in_use）再加 3（湊成對子、刻子時多台）；孤張金牌只加 1，兩張孤張字牌中留金牌，
    但不會為了它打掉孤張中張（實戰 10/2 14:07 向聽、進張都一樣時，留著孤張金牌北、打掉六萬）。"""
    bonus = (3 if gold_in_use(hand, tile) else 1) if tile in gold_tiles else 0
    value = hand[tile] * 6 + bonus
    if tile >= 27:
        return value - 2
    rank = tile % 9
    for offset, weight in ((1, 3), (2, 1)):
        for neighbor in (rank - offset, rank + offset):
            if 0 <= neighbor <= 8:
                value += weight * min(hand[tile - rank + neighbor], 2)
    return value - (1 if rank in (0, 8) else 0)


GOLD_DISCARD_PENALTY = 8
"""打出一張金牌的代價，以有效進張張數計（由模擬器比較決定，見 CLAUDE.md）。"""

def gold_in_use(hand: Sequence[int], tile: int) -> bool:
    """這張牌有沒有機會組進牌型：同種 2 張以上，或（數牌）前後兩格內有牌。"""
    if hand[tile] >= 2:
        return True
    if tile >= 27:
        return False
    rank = tile % 9
    return any(0 <= rank + offset <= 8 and hand[tile + offset]
               for offset in (-2, -1, 1, 2))


SWAP_MIN_GAIN = 0.02
"""期望向聽數至少要降低這麼多才值得換。"""


def choose_swap_tiles(hand: Sequence[int], limit: int = MAX_SWAP,
                      gold_tiles: Sequence[int] = (),
                      valued_honors: Sequence[int] = ()) -> tuple[int, ...]:
    """換三張：換掉「換一張隨機新牌後，期望向聽數比現在低」的牌，最多 limit 張。

    新牌的機率依自己看不到的張數（4 - 手上張數）估計。各張分開評估，
    取期望向聽數最低的幾張；沒有值得換的就不換。對子多時向聽數由嚦咕嚦咕
    決定，所有單張的期望值都一樣，此時再以一般牌型的期望向聽數區分
    （例如孤張字牌比能搭子的數牌更該換）。金牌留著胡牌時每張 +1 台，不換。
    會加台的字牌（valued_honors，見 honor_tai）成對以上也不換：向聽數只把它當普通對子，
    實戰 10/2 12:57 曾把東風圈的一對東換掉一張。
    """
    gold = set(gold_tiles)
    valued = set(valued_honors)
    hand = list(hand)
    current = shanten(hand, 0)
    unseen = [4 - count for count in hand]
    candidates = []
    for tile, count in enumerate(hand):
        if not count or tile in gold or (tile in valued and count >= 2):
            continue
        hand[tile] -= 1
        total = weighted = weighted_standard = 0
        for new_tile, copies in enumerate(unseen):
            if copies <= 0 or hand[new_tile] >= 4:
                continue
            hand[new_tile] += 1
            weighted += copies * shanten(hand, 0)
            weighted_standard += copies * standard_shanten(hand, 0)
            total += copies
            hand[new_tile] -= 1
        hand[tile] += 1
        # 換出去的牌別人可能拿到：中張最好用，要自己明顯變好才值得送出去
        expected = weighted / total + swap_gift_cost(tile)
        if expected < current - SWAP_MIN_GAIN:
            candidates.append((round(expected, 6), weighted_standard / total, tile))
    return tuple(tile for *_, tile in sorted(candidates)[:limit])


def honor_tai(state: GameState, player: int, tile: int) -> int:
    """這種字牌碰成刻子能加幾台（game.scoring 的三元牌、圈風牌、門風牌）；莊家不明時門風不算。"""
    if tile >= 31:
        return 1
    tai = int(tile == state.round_wind)
    if state.known_dealer is not None and tile == state.seat_winds[player]:
        tai += 1
    return tai


def swap_gift_cost(tile: int) -> float:
    """換出這張牌的代價（以期望向聽數計）：對別人越好用的牌越高。
    中張 3～7 最容易組順子，2、8 次之，1、9 只能組邊張，字牌只能碰。"""
    if tile >= 27:
        return 0.0
    rank = tile % 9 + 1
    if rank in (1, 9):
        return 0.03
    if rank in (2, 8):
        return 0.08
    return 0.15


def hand_value(hand: Sequence[int], n_open_melds: int) -> tuple[int, int]:
    """(向聽數, -有效進張張數)，越小越好；有效進張以自己手上沒有的張數計。"""
    hand = list(hand)
    outs = sum(4 - hand[tile] for tile in effective_tiles(hand, n_open_melds))
    return shanten(hand, n_open_melds), -outs


def claim_value(hand: Sequence[int], n_open_melds: int, action: Action,
                claimed: int | None) -> tuple[int, int]:
    """吃碰槓之後的 hand_value；吃碰要接著打一張，取打掉可打的牌後最好的結果。
    明槓之後要補牌，只看向聽數。"""
    assert claimed is not None
    concealed = list(hand)
    used = list(action.tiles)
    used.remove(claimed)
    for tile in used:
        concealed[tile] -= 1
    if action.kind == ActionType.KONG:
        return shanten(concealed, n_open_melds + 1), 0
    options = []
    for tile in allowed_discards(concealed, forbidden_after_claim(action, claimed)):
        concealed[tile] -= 1
        options.append((shanten(concealed, n_open_melds + 1), tile))
        concealed[tile] += 1
    lowest = min(value for value, _ in options)
    best = None
    for value, tile in options:
        if value != lowest:
            continue
        concealed[tile] -= 1
        candidate = hand_value(concealed, n_open_melds + 1)
        concealed[tile] += 1
        best = candidate if best is None else min(best, candidate)
    assert best is not None
    return best


def kong_is_worth(hand: Sequence[int], n_open_melds: int, tile: int,
                  forbidden: Sequence[int] = ()) -> bool:
    """輪到自己時槓 tile（手上 4 張 → 暗槓；否則是把 1 張加到碰過的刻子 → 加槓）划不划算：
    槓完的向聽數不比最好的打法差就槓（槓牌有台、多摸一張、多翻一種金牌）。"""
    after = list(hand)
    if after[tile] >= 4:
        after[tile] -= 4
        melds = n_open_melds + 1
    else:
        after[tile] -= 1
        melds = n_open_melds
    best = min(shanten(_without(hand, t), n_open_melds)
               for t in allowed_discards(list(hand), tuple(forbidden)))
    return shanten(after, melds) <= best


def _without(hand: Sequence[int], tile: int) -> list[int]:
    result = list(hand)
    result[tile] -= 1
    return result


def choose_self_kong(state: GameState, player: int) -> Action | None:
    """輪到自己時要不要暗槓或加槓。"""
    me = state.players[player]
    for action in legal_actions(state, player):
        if action.kind == ActionType.KONG and kong_is_worth(me.hand, me.open_melds, action.tile):
            return action
    return None


def visible_tile_counts(state: GameState, player: int) -> list[int]:
    """以個人視角統計自己的手牌、牌河與公開副露。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    visible = [0] * 34
    own = state.players[player]
    for tile, count in enumerate(own.hand):
        visible[tile] += count
    for known_player in state.players:
        for tile in known_player.discards:
            visible[tile] += 1
        for meld in known_player.melds:
            for tile in meld.tiles:
                visible[tile] += 1
    if any(count > 4 for count in visible):
        raise ValueError("牌局狀態出現超過 4 張同種牌")
    return visible


def safe_tiles(state: GameState, player: int, use_declared: bool = True) -> set[int]:
    """回傳對可觀測威脅對手都已打過的現物牌。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    opponents = [
        opponent_state for opponent, opponent_state in enumerate(state.players)
        if opponent != player and (len(opponent_state.melds) >= 2
                                   or (use_declared and state.declared[opponent]))
    ]
    if not opponents:
        opponents = [
            opponent_state for opponent, opponent_state in enumerate(state.players)
            if opponent != player
        ]
    safe: set[int] | None = None
    for opponent_state in opponents:
        discards = set(opponent_state.discards)
        safe = discards if safe is None else safe & discards
    return safe or set()


def suji_tiles(state: GameState, player: int) -> set[int]:
    """回傳筋牌候選；這不是絕對安全牌，僅降低兩面聽風險。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    discarded = set()
    for opponent, opponent_state in enumerate(state.players):
        if opponent != player:
            discarded.update(opponent_state.discards)
    suji = set()
    for suit_start in (0, 9, 18):
        for rank in range(1, 10):
            tile = suit_start + rank - 1
            partners = []
            if rank > 3:
                partners.append(suit_start + rank - 4)
            if rank <= 6:
                partners.append(suit_start + rank + 2)
            if any(partner in discarded for partner in partners):
                suji.add(tile)
    return suji


def danger_score(state: GameState, player: int, tile: int) -> float:
    """以公開資訊估計棄牌危險度，分數越低越安全。"""
    if not 0 <= player < 4 or not 0 <= tile < 34:
        raise ValueError("player 必須在 0-3 之間，tile 必須在 0-33 之間")
    safe = safe_tiles(state, player)
    if tile in safe:
        return 0.0
    suji = suji_tiles(state, player)
    if tile in suji:
        return 1.0
    score = 2.0
    if is_honor(tile):
        score += 1.0
    elif is_suited(tile) and tile % 9 in {0, 8}:
        score += 0.5
    visible = visible_tile_counts(state, player)[tile]
    if visible == 0:
        score += 1.0
    return score


def opponent_threat_score(state: GameState, observer: int, opponent: int,
                          use_declared: bool = True) -> int:
    """以公開資訊估計對手威脅，不讀取對手手牌。"""
    if not 0 <= observer < 4 or not 0 <= opponent < 4:
        raise ValueError("observer 與 opponent 必須在 0-3 之間")
    if observer == opponent:
        raise ValueError("observer 與 opponent 不可相同")
    if use_declared and state.declared[opponent]:
        return 5  # 宣告聽牌：確定已經聽了
    opponent_state = state.players[opponent]
    meld_count = len(opponent_state.melds)
    if meld_count >= 3:
        score = 4
    elif meld_count == 2:
        score = 3
    elif meld_count == 1:
        score = 2
    else:
        score = 0
    if meld_count and len(opponent_state.discards) >= 6:
        score += 1
    return min(score, 5)


DECLARED_FOLD_SHANTEN = 2
"""有對手宣告聽牌時，自己向聽數達到這個值就棄胡。"""


def should_fold(state: GameState, player: int, use_declared: bool = True,
                declared_fold_shanten: int | None = None) -> bool:
    """判斷是否值得放棄進攻，進入完全防守。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    own = state.players[player]
    own_shanten = shanten(own.hand, own.open_melds)
    highest_threat = max(
        opponent_threat_score(state, player, opponent, use_declared)
        for opponent in range(4) if opponent != player
    )
    if use_declared and any(state.declared[opponent] for opponent in range(4)
                            if opponent != player):
        if declared_fold_shanten is None:
            declared_fold_shanten = DECLARED_FOLD_SHANTEN
        return own_shanten >= declared_fold_shanten
    return own_shanten >= 2 and highest_threat >= 3


DECLARE_MIN_LIVE = 0
"""聽的牌還剩（看不到的）幾張以上才宣告聽牌；0 = 一律宣告。宣告後手牌鎖住、不能再換成更好的聽，
沒宣告也能胡（沒有振聽），只是少 1 台。第一次打牌就聽（地聽 4 台）一律宣告。"""


def declare_is_worth(state: GameState, player: int, tile: int,
                     min_live: int | None = None) -> bool:
    """打出 tile 後聽牌時，要不要宣告。"""
    if min_live is None:
        min_live = DECLARE_MIN_LIVE
    me = state.players[player]
    if min_live <= 0 or not me.discards:
        return True
    after = list(me.hand)
    after[tile] -= 1
    visible = visible_tile_counts(state, player)
    live = sum(max(0, 4 - visible[wait]) for wait in winning_tiles(after, me.open_melds))
    return live >= min_live


CLAIM_MODE = "improve"
"""吃碰的標準：
- loose：向聽變好，或向聽相同但有效進張變多（原本的做法）
- improve：向聽一定要變好
- near：向聽變好且吃碰完不超過一向聽；碰會加台的字牌（三元、圈風、已知門風）只要向聽變好
實戰 10/1：建議過吃碰的局放槍率 16~21%、平均每局 −23~−106，沒吃碰的局 12%、+165（有選擇偏差）。"""


def claim_is_worth(state: GameState, player: int, action: Action, before: tuple[int, int],
                   after: tuple[int, int], mode: str | None = None) -> bool:
    """吃碰值不值得（明槓另外判斷）。"""
    mode = mode or CLAIM_MODE
    if mode == "loose":
        return after < before
    if after[0] >= before[0]:
        return False
    if mode == "improve":
        return True
    if mode == "near":
        tile = state.last_discard
        valued = action.kind == ActionType.PUNG and tile is not None and tile >= 27 and (
            tile >= 31 or tile == state.round_wind
            or (state.known_dealer is not None and tile == state.seat_winds[player]))
        return valued or after[0] <= 1
    raise ValueError(f"未知的吃碰標準：{mode}")


def choose_rule_action_for_player(
    state: GameState, player: int, rng: Random | None = None,
    gold_penalty: float | None = None, use_declared: bool = True,
    declared_fold_shanten: int | None = None, claim_mode: str | None = None,
) -> Action | None:
    """只替指定玩家做決策，不讀取對手隱藏手牌；gold_penalty 見 choose_discard。
    use_declared=False 時忽略對手宣告聽牌（評估這項資訊的價值用）。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    actions = legal_actions(state, player)
    win = next((action for action in actions if action.kind == ActionType.WIN), None)
    if win is not None:
        return win
    if state.phase == Phase.SWAP:
        valued = [tile for tile in range(27, 34) if honor_tai(state, player, tile)]
        return Action(ActionType.SWAP, tiles=choose_swap_tiles(state.players[player].hand,
                                                               gold_tiles=state.gold_tiles,
                                                               valued_honors=valued))
    if state.phase == Phase.RESPONSE:
        current = state.players[player]
        baseline = hand_value(current.hand, current.open_melds)
        best = None
        for action in actions:
            if action.kind not in {ActionType.CHOW, ActionType.PUNG, ActionType.KONG}:
                continue
            after = claim_value(current.hand, current.open_melds, action, state.last_discard)
            # 吃碰要讓向聽數變好、或向聽相同但有效進張變多才值得（拆掉已成的面子去吃
            # 不划算）；明槓多摸一張，向聽不變差就槓
            if action.kind == ActionType.KONG:
                worth = after[0] <= baseline[0]
                after = (after[0], float("-inf"))  # 向聽相同時槓優先於碰（槓有台、多摸一張）
            else:
                worth = claim_is_worth(state, player, action, baseline, after, claim_mode)
            if worth and (best is None or after < best[0]):
                best = (after, action)
        if best is not None:
            return best[1]
        return next((action for action in actions if action.kind == ActionType.PASS), None)
    if state.phase == Phase.DISCARD:
        if state.declared[player]:
            return actions[0]  # 聽牌後手牌鎖住，只剩打掉摸進的牌
        kong = choose_self_kong(state, player)
        if kong is not None:
            return kong
        defensive = should_fold(state, player, use_declared, declared_fold_shanten)
        tile = choose_discard(state.players[player].hand,
                              state.players[player].open_melds, rng,
                              visible_tile_counts(state, player),
                              safe_tiles(state, player, use_declared),
                              suji_tiles(state, player), defensive=defensive,
                              forbidden=state.forbidden_discards if state.claimed else (),
                              gold_tiles=state.gold_tiles, gold_penalty=gold_penalty)
        declare = Action(ActionType.DECLARE, tile=tile)
        if not defensive and declare in actions and declare_is_worth(state, player, tile):
            return declare  # 聽牌 +1 台；聽的牌太少時先不宣告，留著換更好的聽
        return Action(ActionType.DISCARD, tile=tile)
    action = next((action for action in actions if action.kind == ActionType.DRAW), None)
    return action


def choose_rule_action(state: GameState, rng: Random | None = None) -> tuple[int, Action] | None:
    """模擬器相容的聚合器；實際玩家決策請使用指定座位 API。"""
    players = state.response_players if state.phase == Phase.RESPONSE else [state.current_player]
    choices = [(player, choose_rule_action_for_player(state, player, rng)) for player in players]
    choices = [(player, action) for player, action in choices if action is not None]
    # 同時回應時：胡 > 槓/碰 > 吃 > 其他；同優先權依座位順序
    for kinds in ({ActionType.WIN}, {ActionType.KONG, ActionType.PUNG}, {ActionType.CHOW}):
        claims = [(player, action) for player, action in choices if action.kind in kinds]
        if claims:
            return claims[0]
    active = [(player, action) for player, action in choices if action.kind != ActionType.PASS]
    if active:
        return active[0]
    return choices[0] if choices else None