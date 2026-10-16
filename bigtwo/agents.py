"""大老二基準 AI：隨機、出最小、規則式（拆牌規劃）。"""
from __future__ import annotations

from random import Random

from .cards import CLUB_THREE, NUM_RANKS, RANK_TWO, rank_of, suit_of
from .combos import STRAIGHTS, Combo, Kind, beats, classify
from .game import View


def random_policy(view: View, rng: Random) -> Combo | None:
    options: list[Combo | None] = list(view.moves)
    if view.can_pass:
        options.append(None)
    return rng.choice(options)


def smallest_policy(view: View, rng: Random) -> Combo | None:
    """能出就出最小的（帶頭時出含最小那張的單張）。"""
    return min(view.moves, key=_strength) if view.moves else None


def _strength(combo: Combo) -> tuple:
    return (combo.size == 5, combo.kind, combo.key)


# ---------- 規則式：把手牌拆成最少手數的組合 ----------

def _five_card_candidates(hand: list[int]) -> list[Combo]:
    """拆牌用的五張組合：每種點數組合只取一種挑法（盡量用落單、花色小的牌），避免組合爆炸。"""
    groups = [[] for _ in range(NUM_RANKS)]
    for c in sorted(hand):
        groups[rank_of(c)].append(c)
    out = []
    for ranks in STRAIGHTS:
        if all(groups[r] for r in ranks):
            plain = [min(groups[r], key=lambda c: (len(groups[rank_of(c)]) > 1, c)) for r in ranks]
            out.append(classify(plain))
            # 同花順另外試
            for suit in range(4):
                flush = [next((c for c in groups[r] if suit_of(c) == suit), None) for r in ranks]
                if None not in flush:
                    out.append(classify(flush))
    pairs = [g[:2] for g in groups if len(g) >= 2]
    for r, g in enumerate(groups):
        if len(g) >= 3:
            for p in pairs:
                if rank_of(p[0]) != r:
                    out.append(classify(g[:3] + p))
                    break
        if len(g) == 4:
            others = sorted((c for c in hand if rank_of(c) != r),
                            key=lambda c: (len(groups[rank_of(c)]) > 1, c))
            if others:
                out.append(classify(g + others[:1]))
    return [c for c in out if c is not None]


def _small_groups(hand: list[int]) -> list[Combo]:
    groups = [[] for _ in range(NUM_RANKS)]
    for c in sorted(hand):
        groups[rank_of(c)].append(c)
    out = []
    for g in groups:
        if len(g) == 4:
            out += [classify(g[:2]), classify(g[2:])]
        elif g:
            out.append(classify(g))
    return out


def plan_hand(hand) -> list[Combo]:
    """拆成手數最少的組合；同手數時落單小牌越少越好。"""
    hand = sorted(hand)
    best: tuple | None = None

    def score(combos: list[Combo]) -> tuple:
        low_singles = sum(1 for c in combos if c.kind == Kind.SINGLE and rank_of(c.cards[0]) < 8)
        return (len(combos), low_singles)

    def search(rest: list[int], chosen: list[Combo], depth: int):
        nonlocal best
        combos = chosen + _small_groups(rest)
        s = score(combos)
        if best is None or s < best[0]:
            best = (s, combos)
        if depth == 2:
            return
        for five in _five_card_candidates(rest):
            if chosen and _strength(five) < _strength(chosen[-1]):
                continue
            left = list(rest)
            for c in five.cards:
                left.remove(c)
            search(left, chosen + [five], depth + 1)

    search(hand, [], 0)
    return best[1]


def _beatable_by(combo: Combo, unseen: list[int]) -> bool:
    """看不到的牌裡有沒有同牌型更大的（只算單張、對子、三條；五張牌型與炸彈另判）。"""
    if combo.kind == Kind.SINGLE:
        return any(c > combo.cards[0] for c in unseen)
    groups = [[] for _ in range(NUM_RANKS)]
    for c in unseen:
        groups[rank_of(c)].append(c)
    if combo.kind == Kind.PAIR:
        return any(len(g) >= 2 and max(g) > combo.key[0] for g in groups)
    if combo.kind == Kind.TRIPLE:
        return any(len(g) >= 3 and r > combo.key[0] for r, g in enumerate(groups))
    return True


def outlook(view: View, plan: list[Combo]) -> tuple[int, int]:
    """（控制牌數, 小牌數）。控制＝別人很難壓的一手（2、別人壓不過的單張／對子／三條、葫蘆以上、
    10 起頭以上的順子）；小牌＝8 以下的單張與對子（要靠上手才出得去）。依攻略「數控制與小牌」判斷好壞牌。"""
    seen = view.played | set(view.hand)
    unseen = [c for c in range(52) if c not in seen]
    control = small = 0
    for combo in plan:
        if combo.size == 5:
            control += combo.kind >= Kind.FULL_HOUSE or combo.key[0] >= 7
        elif not _beatable_by(combo, unseen) or rank_of(combo.cards[-1]) == RANK_TWO:
            control += 1
        elif combo.kind in (Kind.SINGLE, Kind.PAIR) and rank_of(combo.cards[-1]) <= 5:
            small += 1
    return control, small


ESCAPE_MARGIN = 1
"""控制牌 + ESCAPE_MARGIN ≤ 小牌 就算爛牌。arena 6000 副（對 margin 2）：0 +0.13±0.12、1 +0.24±0.09、
3 −0.26±0.06、一律逃牌 −0.21±0.16。"""
ESCAPE_LIMIT = 9
"""逃牌時手上超過幾張就拆牌也要出（輸家超過 9 張再乘倍率）。7 張 −0.06±0.05、11 張 −0.18±0.04。"""


FINISH_CARDS = 6
"""手上剩這麼多張以下時，先找「每一手別人都壓不過、最後一手出完」的路線（攻略：剩 3～5 張時算好能不能一路出完）。
arena 8000 副對沒有殘局路線：+0.37±0.06 張/局（勝率 25.0%→25.6%）。"""


def _bombs_exist(unseen: list[int]) -> bool:
    groups = [[] for _ in range(NUM_RANKS)]
    for c in unseen:
        groups[rank_of(c)].append(c)
    if any(len(g) == 4 for g in groups):
        return True
    for suit in range(4):
        ranks = {rank_of(c) for c in unseen if suit_of(c) == suit}
        if any(all(r in ranks for r in seq) for seq in STRAIGHTS):
            return True
    return False


def unbeatable(combo: Combo, unseen: list[int], others_cards: int = 13) -> bool:
    """看不到的牌全部加起來也組不出能壓 combo 的牌（保守：當成同一家拿到全部）。"""
    if combo.size < 5:
        # 不算鐵支／同花順：看不到的牌那麼多，幾乎總湊得出一副，算進去就沒有任何一手「壓不過」
        return not _beatable_by(combo, unseen)
    from .combos import all_combos, beats
    if others_cards < 5:
        return True
    return not any(m.size == 5 and beats(m, combo) for m in all_combos(unseen))


def finish_route(hand, unseen: list[int], others_cards: int) -> list[Combo] | None:
    """把手牌拆成「前面每手都壓不過、最後一手隨意」的出法；找不到回傳 None。"""
    whole = classify(list(hand))
    if whole is not None:
        return [whole]
    from .combos import all_combos
    for combo in sorted(all_combos(list(hand)), key=lambda c: (-c.size, _strength(c))):
        if unbeatable(combo, unseen, others_cards):
            rest = [c for c in hand if c not in combo.cards]
            route = finish_route(rest, unseen, others_cards)
            if route is not None:
                return [combo] + route
    return None


def _finish(view: View) -> Combo | None:
    if len(view.hand) > FINISH_CARDS:
        return None
    seen = view.played | set(view.hand)
    unseen = [c for c in range(52) if c not in seen]
    others = max(n for s, n in enumerate(view.counts) if s != view.seat)
    moves = set(view.moves)
    if view.last is None:
        route = finish_route(view.hand, unseen, others)
        return route[0] if route and route[0] in moves else None
    for m in sorted(view.moves, key=lambda c: -c.size):
        rest = [c for c in view.hand if c not in m.cards]
        if not rest:
            return m
        if unbeatable(m, unseen, others) and finish_route(rest, unseen, others) is not None:
            return m
    return None


def make_rule_policy(escape: bool = True, finish: bool = True):
    """escape：爛牌時改成逃牌——張數超過 9 張時帶頭出一次最多張的那手、跟牌時拆牌也要出，
    先降到 9 張以下（輸家「超過9張」「13張未出」各再乘本局倍率；攻略「少輸比贏牌重要」）。
    arena 8000 副（margin 2、會提早打 2 的版本）對不逃牌：+0.50±0.07 張/局。
    提早把 2 打掉（攻略說 2 不要留到最後）實測反而差：不刻意打 2 +0.07±0.05，所以不做。"""

    def policy(view: View, rng: Random) -> Combo | None:
        if not view.moves:
            return None
        if finish:
            route_move = _finish(view)
            if route_move is not None:
                return route_move
        plan = plan_hand(view.hand)
        if escape:
            control, small = outlook(view, plan)
            if control + ESCAPE_MARGIN <= small:
                choice = _escape(view, plan)
                if choice is not None:
                    return choice
        return _normal(view, plan)

    return policy


def _escape(view: View, plan: list[Combo]) -> Combo | None:
    moves = set(view.moves)
    many = len(view.hand) > ESCAPE_LIMIT
    if view.last is None:
        legal_plan = [c for c in plan if c in moves] or list(view.moves)
        if many:   # 一次出最多張的那手（五張牌型、三條、對子）
            return max(legal_plan, key=lambda c: (c.size, -_strength(c)[2][0] if c.size < 5 else 0))
        return None    # 張數已安全：照一般打法
    in_plan = sorted((c for c in plan if c in moves), key=_strength)
    if in_plan:
        return in_plan[0]
    if many:               # 還超過 9 張：拆牌也要出
        return max(view.moves, key=lambda c: (c.size, -_strength(c)[2][0] if c.size < 5 else 0))
    return None


def _normal(view: View, plan: list[Combo]) -> Combo | None:
    moves = set(view.moves)
    nxt = view.counts[(view.seat + 1) % 4]
    danger = min(n for s, n in enumerate(view.counts) if s != view.seat)

    if view.last is None:
        legal_plan = [c for c in plan if c in moves]
        if len(plan) == 1 and legal_plan:
            return legal_plan[0]
        if len(plan) == 2 and len(legal_plan) == 2:   # 先出比較大的那手拿回出牌權，留最後一手
            return max(legal_plan, key=_strength)
        pool = legal_plan or list(view.moves)
        if nxt == 1:         # 下家剩一張：不要帶小單張
            non_single = [c for c in pool if c.kind != Kind.SINGLE]
            if non_single:
                return min(non_single, key=_strength)
            return max(view.moves, key=lambda c: (c.kind == Kind.SINGLE, c.key))
        # 帶含最小牌的那一手；五張牌型優先（一次出多張）
        lowest = min(view.hand)
        with_lowest = [c for c in pool if lowest in c.cards]
        if with_lowest:
            return max(with_lowest, key=lambda c: (c.size, -c.key[0] if c.size < 5 else 0))
        return min(pool, key=_strength)

    # 跟牌
    in_plan = sorted((c for c in plan if c in moves), key=_strength)
    if nxt == 1 and view.last.kind == Kind.SINGLE:
        return max(view.moves, key=_strength)
    if in_plan:
        choice = in_plan[0]
        # 別人還多張時，不急著用 2 壓小牌
        if (choice.kind == Kind.SINGLE and rank_of(choice.cards[0]) == RANK_TWO
                and danger > 4 and len(plan) > 3 and rank_of(view.last.cards[-1]) < 8):
            return None
        return choice
    if danger <= 3 or len(view.hand) <= 4:
        return min(view.moves, key=_strength)
    return None



rule_policy = make_rule_policy(escape=True)
old_rule_policy = make_rule_policy(escape=False, finish=False)     # 10/17 之前（沒有逃牌、殘局路線）
nofinish_policy = make_rule_policy(escape=True, finish=False)


SWAP_MAX_RANK = 8
"""換三張只換 J 以下的落單牌。arena 2000 副（其他三家都換）：J 以下比 8 以下 +1.01±0.42 張；
Q、K、A 以下 +0.03、−0.03、−0.27，10 以下 −0.21；連梅花 3 一起換 −0.18±0.16。"""


def rule_swap(hand, rng: Random) -> list[int]:
    """換三張：拆牌規劃裡落單的小牌（J 以下、不含梅花 3）由小到大最多 3 張。"""
    singles = sorted(c.cards[0] for c in plan_hand(hand) if c.kind == Kind.SINGLE)
    return [c for c in singles if rank_of(c) <= SWAP_MAX_RANK and c != CLUB_THREE][:3]
