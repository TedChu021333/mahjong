"""台灣 16 張麻將台數計算。

由胡牌結構算得出的牌型在這裡判斷；槓上開花、海底撈月、河底撈魚、搶槓
這類對局事件由 simulator 在事件發生時以 events 傳入。台數值集中在 TAI，
依實際遊戲的台數表調整即可。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .rules import Meld
from .tiles import NUM_TILE_TYPES, is_flower, is_honor, is_suited
from .win import decompose, is_lickgu, is_win, winning_tiles


TAI = {
    "莊家": 1,
    "門清": 1,
    "自摸": 1,
    "門清自摸": 3,
    "全求人": 2,
    "平胡": 2,
    "獨聽": 1,
    "碰碰胡": 4,
    "三暗刻": 2,
    "四暗刻": 5,
    "五暗刻": 8,
    "混一色": 4,
    "清一色": 8,
    "字一色": 16,
    "三元牌": 1,
    "小三元": 4,
    "大三元": 8,
    "門風牌": 1,
    "圈風牌": 1,
    "小四喜": 8,
    "大四喜": 16,
    "花牌": 1,
    "金牌": 1,
    "胡金牌": 3,
    "聽牌": 1,
    "八仙過海": 8,
    "七搶一": 8,
    "槓上開花": 1,
    "海底撈月": 1,
    "河底撈魚": 1,
    "搶槓": 1,
    "嚦咕嚦咕": 8,
}
EVENTS = ("槓上開花", "海底撈月", "河底撈魚", "搶槓")
DRAGONS = range(31, 34)
WINDS = range(27, 31)


@dataclass(frozen=True)
class Score:
    tai: int
    patterns: tuple[str, ...]
    pair: int | None = None
    melds: tuple[tuple[str, int], ...] = ()


def _validate_counts(counts: Sequence[int]) -> list[int]:
    if len(counts) != NUM_TILE_TYPES:
        raise ValueError("手牌計數陣列必須長度為 34")
    if any(count < 0 or count > 4 for count in counts):
        raise ValueError("每種牌的數量必須在 0-4 之間")
    return list(counts)


def _validate_melds(melds: Sequence[Meld]) -> None:
    for meld in melds:
        if meld.kind not in {"chow", "pung", "kong"}:
            raise ValueError(f"未知副露種類：{meld.kind}")


def _flower_patterns(flowers: Sequence[int]) -> list[str]:
    """明星三缺一：每張花牌 1 台，不分正花（見 train/遊戲流程.mp4 976 秒的結算）。"""
    if any(not is_flower(tile) for tile in flowers) or len(set(flowers)) != len(flowers):
        raise ValueError("花牌必須是不重複的 34-41")
    return ["花牌"] * len(flowers)


def _gold_patterns(hand: Sequence[int], open_melds: Sequence[Meld],
                   gold_tiles: Sequence[int], win_tile: int | None) -> list[str]:
    """明星三缺一金牌：手上（含副露）每張金牌 +1 台，胡的那張是金牌再 +3 台。

    「胡金牌」是否也計入收集的 +1 台，遊戲說明沒寫清楚，這裡兩者都算，
    之後以結算畫面校正。
    """
    gold = set(gold_tiles)
    held = sum(hand[tile] for tile in gold)
    held += sum(1 for meld in open_melds for tile in meld.tiles if tile in gold)
    patterns = ["金牌"] * held
    if win_tile is not None and win_tile in gold:
        patterns.append("胡金牌")
    return patterns


def _context_patterns(
    self_draw: bool, flowers: Sequence[int], seat_wind: int | None,
    dealer: bool, events: Sequence[str], declared: bool,
) -> list[str]:
    patterns = []
    if dealer:
        patterns.append("莊家")
    if declared:
        patterns.append("聽牌")
    patterns.extend(_flower_patterns(flowers))
    for event in events:
        if event not in EVENTS:
            raise ValueError(f"未知的對局事件：{event}")
        if event in {"槓上開花", "海底撈月"} and not self_draw:
            raise ValueError(f"{event} 必須是自摸")
        if event in {"河底撈魚", "搶槓"} and self_draw:
            raise ValueError(f"{event} 必須是胡別人的牌")
        patterns.append(event)
    return patterns


def _is_two_sided(start: int, win_tile: int) -> bool:
    """以 win_tile 完成順子 start..start+2 時是否為兩面聽（非邊張、中洞）。"""
    if win_tile == start:
        return start % 9 != 6      # 789 胡 7 是邊張
    if win_tile == start + 2:
        return start % 9 != 0      # 123 胡 3 是邊張
    return False


def _patterns_for_decomposition(
    hand: Sequence[int],
    open_melds: Sequence[Meld],
    pair: int,
    concealed_melds: Sequence[tuple[str, int]],
    self_draw: bool,
    has_flowers: bool,
    seat_wind: int | None,
    round_wind: int | None,
    win_tile: int | None,
    single_wait: bool,
) -> list[str]:
    exposed = [meld for meld in open_melds if meld.from_player is not None]
    menzen = not exposed
    sets = [(kind, tile) for kind, tile in concealed_melds]
    sets.extend(("chow" if meld.kind == "chow" else "pung", meld.tiles[0]) for meld in open_melds)
    tiles_used = [tile for tile, count in enumerate(hand) if count]
    tiles_used.extend(tile for meld in open_melds for tile in meld.tiles)
    patterns: list[str] = []

    if menzen and self_draw:
        patterns.append("門清自摸")
    elif menzen:
        patterns.append("門清")
    elif self_draw:
        patterns.append("自摸")
    if len(exposed) == 5 and not self_draw:
        patterns.append("全求人")

    if all(kind == "pung" for kind, _ in sets):
        patterns.append("碰碰胡")
    elif (all(kind == "chow" for kind, _ in sets) and not is_honor(pair)
          and not has_flowers and not self_draw and win_tile is not None
          and not single_wait
          and any(kind == "chow" and start <= win_tile <= start + 2
                  and _is_two_sided(start, win_tile) for kind, start in concealed_melds)):
        patterns.append("平胡")
    if single_wait:
        patterns.append("獨聽")

    # 暗刻：手中的刻子與暗槓；胡別人的牌完成的刻子算明刻，但胡的那張若
    # 也能歸到同一拆法的對子或順子，就視為落在那裡（取有利解釋）。
    concealed_pungs = [tile for kind, tile in concealed_melds if kind == "pung"]
    if not self_draw and win_tile in concealed_pungs:
        elsewhere = pair == win_tile or any(
            kind == "chow" and start <= win_tile <= start + 2 for kind, start in concealed_melds
        )
        if not elsewhere:
            concealed_pungs.remove(win_tile)
    hidden = len(concealed_pungs) + sum(
        1 for meld in open_melds if meld.kind == "kong" and meld.from_player is None
    )
    if hidden >= 3:
        patterns.append({3: "三暗刻", 4: "四暗刻", 5: "五暗刻"}[hidden])

    suits = {tile // 9 for tile in tiles_used if is_suited(tile)}
    if not suits:
        patterns.append("字一色")
    elif len(suits) == 1:
        patterns.append("混一色" if any(is_honor(tile) for tile in tiles_used) else "清一色")

    honor_pungs = [tile for kind, tile in sets if kind == "pung" and is_honor(tile)]
    dragon_pungs = [tile for tile in honor_pungs if tile in DRAGONS]
    if len(dragon_pungs) == 3:
        patterns.append("大三元")
    elif len(dragon_pungs) == 2 and pair in DRAGONS:
        patterns.append("小三元")
    else:
        patterns.extend("三元牌" for _ in dragon_pungs)
    wind_pungs = [tile for tile in honor_pungs if tile in WINDS]
    if len(wind_pungs) == 4:
        patterns.append("大四喜")
    elif len(wind_pungs) == 3 and pair in WINDS:
        patterns.append("小四喜")
    else:
        for tile in wind_pungs:
            if tile == seat_wind:
                patterns.append("門風牌")
            if tile == round_wind:
                patterns.append("圈風牌")
    return patterns


def _make_score(patterns: Sequence[str], dealer_streak: int,
                pair: int | None = None, melds: tuple[tuple[str, int], ...] = ()) -> Score:
    names = list(patterns)
    tai = sum(TAI[name] for name in names)
    if dealer_streak:
        names.append(f"連{dealer_streak}拉{dealer_streak}")
        tai += 2 * dealer_streak
    return Score(tai, tuple(names) or ("基本胡",), pair, melds)


def score_hand(
    counts: Sequence[int],
    open_melds: Sequence[Meld] = (),
    flowers: Sequence[int] = (),
    self_draw: bool = False,
    allow_lickgu: bool = True,
    seat_wind: int | None = None,
    round_wind: int | None = None,
    win_tile: int | None = None,
    dealer: bool = False,
    dealer_streak: int = 0,
    events: Sequence[str] = (),
    gold_tiles: Sequence[int] = (),
    declared: bool = False,
) -> Score:
    """計算胡牌台數；多種拆法時選台數最高者。

    counts 為胡牌後的暗手（含胡的那張）。win_tile 用來判斷平胡的兩面聽、
    獨聽與胡別人的牌完成的刻子；不提供時這些與聽牌方式有關的牌型不計。
    dealer 表示莊家有參與這次胡牌（莊家胡或莊家放槍），dealer_streak 為連莊數。
    gold_tiles 為本局金牌牌種，declared 表示胡牌者有宣告聽牌。
    """
    hand = _validate_counts(counts)
    _validate_melds(open_melds)
    if dealer_streak < 0 or (dealer_streak and not dealer):
        raise ValueError("連莊數必須搭配莊家")
    if not is_win(hand, len(open_melds), allow_lickgu):
        raise ValueError("不是合法胡牌")
    if win_tile is not None and not hand[win_tile]:
        raise ValueError("胡的那張必須在手牌中")
    context = _context_patterns(self_draw, flowers, seat_wind, dealer, events, declared)
    context += _gold_patterns(hand, open_melds, gold_tiles, win_tile)

    if allow_lickgu and is_lickgu(hand, len(open_melds)):
        patterns = ["嚦咕嚦咕"] + (["自摸"] if self_draw else []) + context
        return _make_score(patterns, dealer_streak)

    single_wait = False
    if win_tile is not None:
        before = hand.copy()
        before[win_tile] -= 1
        single_wait = len(winning_tiles(before, len(open_melds), allow_lickgu)) == 1

    scores = []
    for pair, concealed_melds in decompose(hand):
        patterns = _patterns_for_decomposition(
            hand, open_melds, pair, concealed_melds, self_draw, bool(flowers),
            seat_wind, round_wind, win_tile, single_wait,
        )
        scores.append(_make_score(patterns + context, dealer_streak, pair, concealed_melds))
    if not scores:
        raise ValueError("胡牌拆解失敗")
    return max(scores, key=lambda score: score.tai)


def score_flower_win(kind: str, dealer: bool = False, dealer_streak: int = 0) -> Score:
    """八仙過海、七搶一不看手牌，直接成立。"""
    if kind not in {"八仙過海", "七搶一"}:
        raise ValueError(f"未知的花牌胡：{kind}")
    return _make_score([kind] + (["莊家"] if dealer else []), dealer_streak)
