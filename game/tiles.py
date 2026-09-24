"""台灣16張麻將：牌的編碼與基本工具

編碼（整數）：
    0-8   一萬 ~ 九萬
    9-17  一筒 ~ 九筒
    18-26 一條 ~ 九條
    27-30 東 南 西 北
    31-33 中 發 白
    34-41 花牌：春 夏 秋 冬 梅 蘭 竹 菊

字串表示法（方便寫測試）：
    "123m456p789s1234567z12f"
    m=萬 p=筒 s=條 z=字牌(1-7 = 東南西北中發白) f=花牌(1-8)
"""
from __future__ import annotations

import random

NUM_TILE_TYPES = 34          # 不含花牌的牌種數
FLOWER_START = 34
NUM_FLOWERS = 8
COPIES_PER_TILE = 4

SUIT_NAMES = ["萬", "筒", "條"]
HONOR_NAMES = ["東", "南", "西", "北", "中", "發", "白"]
FLOWER_NAMES = ["春", "夏", "秋", "冬", "梅", "蘭", "竹", "菊"]
CN_NUM = "一二三四五六七八九"


def is_suited(t: int) -> bool:
    return 0 <= t < 27


def is_honor(t: int) -> bool:
    return 27 <= t < NUM_TILE_TYPES


def is_flower(t: int) -> bool:
    return FLOWER_START <= t < FLOWER_START + NUM_FLOWERS


def suit_of(t: int) -> int:
    """0=萬 1=筒 2=條（僅限數牌）"""
    if not is_suited(t):
        raise ValueError(f"{t} 不是數牌")
    return t // 9


def rank_of(t: int) -> int:
    """數字 1-9（僅限數牌）"""
    if not is_suited(t):
        raise ValueError(f"{t} 不是數牌")
    return t % 9 + 1


def tile_name(t: int) -> str:
    if is_suited(t):
        return CN_NUM[t % 9] + SUIT_NAMES[t // 9]
    if is_honor(t):
        return HONOR_NAMES[t - 27]
    if is_flower(t):
        return FLOWER_NAMES[t - FLOWER_START]
    raise ValueError(f"無效的牌編碼：{t}")


def parse(s: str) -> list[int]:
    """把 "123m11z" 這類字串轉成牌編碼列表"""
    tiles: list[int] = []
    pending: list[int] = []
    for ch in s.replace(" ", ""):
        if ch.isdigit():
            pending.append(int(ch))
            continue
        if ch in "mps":
            base, lo, hi = "mps".index(ch) * 9, 1, 9
        elif ch == "z":
            base, lo, hi = 27, 1, 7
        elif ch == "f":
            base, lo, hi = FLOWER_START, 1, 8
        else:
            raise ValueError(f"無法辨識的字元：{ch!r}")
        if not pending:
            raise ValueError(f"'{ch}' 前面沒有數字")
        for d in pending:
            if not lo <= d <= hi:
                raise ValueError(f"{d}{ch} 超出範圍")
            tiles.append(base + d - 1)
        pending = []
    if pending:
        raise ValueError("字串結尾缺少花色字母")
    return tiles


def tile_code(t: int) -> str:
    """單張牌的字串代碼，parse() 的反函數：如 4 → "5m"、27 → "1z"。"""
    if is_suited(t):
        return f"{t % 9 + 1}{'mps'[t // 9]}"
    if is_honor(t):
        return f"{t - 26}z"
    if is_flower(t):
        return f"{t - FLOWER_START + 1}f"
    raise ValueError(f"無效的牌編碼：{t}")


def to_counts(tiles: list[int]) -> list[int]:
    """牌列表 → 長度 34 的計數陣列（花牌不可放進手牌計數）"""
    counts = [0] * NUM_TILE_TYPES
    for t in tiles:
        if is_flower(t):
            raise ValueError("花牌不應出現在手牌計數中，請先補花")
        counts[t] += 1
        if counts[t] > COPIES_PER_TILE:
            raise ValueError(f"{tile_name(t)} 超過 4 張")
    return counts


def counts_to_tiles(counts: list[int]) -> list[int]:
    return [t for t, c in enumerate(counts) for _ in range(c)]


def hand_str(tiles: list[int]) -> str:
    return " ".join(tile_name(t) for t in sorted(tiles))


def build_wall(rng: random.Random | None = None) -> list[int]:
    """洗好的牌牆：136 張一般牌 + 8 張花牌 = 144 張"""
    rng = rng or random.Random()
    wall = [t for t in range(NUM_TILE_TYPES) for _ in range(COPIES_PER_TILE)]
    wall += list(range(FLOWER_START, FLOWER_START + NUM_FLOWERS))
    rng.shuffle(wall)
    return wall
