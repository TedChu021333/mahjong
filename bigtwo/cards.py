"""大老二：牌的編碼

編碼（整數 0-51）＝ 點數 × 4 + 花色，數字越大牌越大（單張直接比整數）：
    點數 0-12：3 4 5 6 7 8 9 10 J Q K A 2
    花色 0-3 ：梅花 方塊 紅心 黑桃（明星三缺一：黑桃 > 紅心 > 方塊 > 梅花）
所以 0 是梅花 3（最小、先出），51 是黑桃 2（最大）。

字串表示法（方便寫測試）：點數 3-9 T J Q K A 2 接花色 c d h s，可用空白分開，
例如 "3c 3d Ts As 2s" 或 "3c3dTsAs2s"；"10" 也可以寫成 T。
"""
from __future__ import annotations

import random

NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4
HAND_SIZE = 13

RANK_CHARS = "3456789TJQKA2"
SUIT_CHARS = "cdhs"
RANK_NAMES = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
SUIT_NAMES = ["梅花", "方塊", "紅心", "黑桃"]
SUIT_SYMBOLS = "♣♦♥♠"

CLUB_THREE = 0
RANK_TWO = 12
RANK_ACE = 11


def card(rank: int, suit: int) -> int:
    return rank * NUM_SUITS + suit


def rank_of(c: int) -> int:
    return c // NUM_SUITS


def suit_of(c: int) -> int:
    return c % NUM_SUITS


def card_name(c: int) -> str:
    return SUIT_NAMES[suit_of(c)] + RANK_NAMES[rank_of(c)]


def card_code(c: int) -> str:
    return RANK_CHARS[rank_of(c)] + SUIT_CHARS[suit_of(c)]


def cards_text(cards) -> str:
    return " ".join(card_code(c) for c in sorted(cards))


def parse_cards(text: str) -> list[int]:
    s = text.replace(" ", "").replace("10", "T")
    if len(s) % 2:
        raise ValueError(f"牌的字串長度不對：{text!r}")
    cards = []
    for i in range(0, len(s), 2):
        r, su = s[i].upper(), s[i + 1].lower()
        if r not in RANK_CHARS or su not in SUIT_CHARS:
            raise ValueError(f"看不懂的牌：{s[i:i + 2]!r}")
        cards.append(card(RANK_CHARS.index(r), SUIT_CHARS.index(su)))
    if len(set(cards)) != len(cards):
        raise ValueError(f"有重複的牌：{text!r}")
    return cards


def deal(rng: random.Random, players: int = 4) -> list[list[int]]:
    """洗牌後每人 13 張（已排序）。"""
    deck = list(range(NUM_CARDS))
    rng.shuffle(deck)
    return [sorted(deck[i * HAND_SIZE:(i + 1) * HAND_SIZE]) for i in range(players)]
