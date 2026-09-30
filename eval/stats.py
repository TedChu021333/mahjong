"""從 result/ 的結算截圖統計實戰戰績（胡牌率、放槍率、每圈與總輸贏）。

    python -m eval.stats                         # 今天全部
    python -m eval.stats --since 18:19 --until 19:53
    python -m eval.stats --date 20260930 --csv result/戰績.csv

每局的小結算畫面（「繼續」）有四列：輸贏金額在右邊，金色為贏、銀色為輸；自己那列的名字是
黃色。金額用 eval/digit_templates/ 的數字模板辨識（字形比對，不需要 OCR 套件）。
一圈的界線：大結算「繼續戰鬥」、倒數後的「再玩一局」、流局後的「再贏一局」截圖。
流局沒有小結算畫面，不會出現在統計裡（輸贏都是 0）。
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from perception.capture import read_image
from perception.continue_button import ContinueButtons

RESULT_DIR = Path("result")
DIGIT_DIR = Path(__file__).with_name("digit_templates")
ROW_TOPS = (65, 272, 480, 688)
ROW_HEIGHT = 75
AMOUNT_X = (1300, 1800)
NAME_BOX = (500, 138, 780, 230)
"""名字所在範圍（相對於列頂，含上下位移的餘裕）。"""
MY_NAME = Path(__file__).with_name("my_name.png")
"""自己的名字（黃字「不沾」）；換帳號時用 --name 指定一張裁好的名字圖。"""
NAME_MIN_SCORE = 0.8
"""實測：自己那列 ≥0.98，其他列 ≤0.55；名字被提示條或卡片蓋住時讀不出來（約 15%）。"""
DIGIT_SIZE = (24, 36)
DIGIT_MIN_IOU = 0.7
ROUND_END = {"繼續戰鬥", "再玩一局", "再贏一局"}
FILE_PATTERN = re.compile(r"^(\d{8})_(\d{6})(?:_\d+)?$")


def text_mask(region: np.ndarray) -> np.ndarray:
    """金額文字（金色或銀色）與藍色底分開：藍色明顯多於紅色的是底色。銀色字有深淺漸層，
    不能用亮度門檻（會斷成好幾塊）。"""
    region = region.astype(int)
    blue, green, red = region[..., 0], region[..., 1], region[..., 2]
    bright = np.maximum(np.maximum(blue, green), red) > 60
    return (((blue - red) < 40) & bright).astype(np.uint8)


def load_digits(directory: Path = DIGIT_DIR) -> list[tuple[str, np.ndarray]]:
    templates = []
    for path in sorted(directory.glob("*.png")):
        mask = read_image(path)[..., 2] > 127
        templates.append((path.stem.split("_")[0], _normalize(mask)))
    return templates


def _normalize(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8) * 255, DIGIT_SIZE, interpolation=cv2.INTER_AREA)
    return resized > 127


@dataclass(frozen=True)
class Row:
    amount: int | None
    """這列的輸贏；沒有金額為 None（0）。"""
    labelled: bool
    """金額前有「胡牌／自摸／放槍」字樣。"""
    is_me: bool


def read_row(image: np.ndarray, top: int, digits: list[tuple[str, np.ndarray]],
             my_name: np.ndarray | None) -> Row:
    left, right = AMOUNT_X
    region = image[top:top + ROW_HEIGHT, left:right]
    mask = text_mask(region)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = sorted((stats[i] for i in range(1, count)
                    if stats[i][3] >= 20 and stats[i][4] >= 40), key=lambda s: s[0])
    text, labelled = "", False
    for x, y, w, h, _ in comps:
        glyph = None
        if h >= 40 and w <= 45:
            shape = _normalize(labels[y:y + h, x:x + w] == index_of(labels, x, y, w, h))
            best = max(digits, key=lambda item: _iou(shape, item[1]))
            if _iou(shape, best[1]) >= DIGIT_MIN_IOU:
                glyph = best[0]
        if glyph is None:
            labelled = labelled or w > 45  # 中文字比數字寬
            text = ""  # 數字要在最右邊連續出現
        elif glyph != "plus":
            text += glyph
    amount = None
    if text:
        pixels = region[mask > 0]
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        golden = float(np.median(hsv[:, 1])) > 90
        amount = int(text) * (1 if golden else -1)
    name_left, name_top, name_right, name_bottom = NAME_BOX
    name = cv2.cvtColor(image[top + name_top:top + name_bottom, name_left:name_right],
                        cv2.COLOR_BGR2GRAY)
    is_me = my_name is not None and         float(cv2.matchTemplate(name, my_name, cv2.TM_CCOEFF_NORMED).max()) >= NAME_MIN_SCORE
    return Row(amount, labelled, is_me)


def index_of(labels: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """外框內最多的元件編號（外框可能包到相鄰字的一角）。"""
    values, counts = np.unique(labels[y:y + h, x:x + w], return_counts=True)
    keep = values > 0
    return int(values[keep][np.argmax(counts[keep])])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / max(1, int((a | b).sum()))


@dataclass(frozen=True)
class Hand:
    stamp: str
    """截圖時間 YYYYMMDD_HHMMSS。"""
    my_amount: int
    outcome: str
    """自摸、胡牌、放槍、被自摸、沒輸贏；讀不出自己那列時為「未知」。"""


def classify(rows: list[Row]) -> tuple[int, str]:
    me = [row for row in rows if row.is_me]
    if len(me) != 1:
        return 0, "未知"
    amount = me[0].amount or 0
    losers = sum(1 for row in rows if row.amount is not None and row.amount < 0)
    if amount > 0:
        return amount, "自摸" if losers == 3 else "胡牌"
    if amount < 0:
        return amount, "被自摸" if losers == 3 else "放槍"
    return 0, "沒輸贏"


@dataclass
class Round:
    hands: list[Hand]

    @property
    def net(self) -> int:
        return sum(hand.my_amount for hand in self.hands)


def collect(directory: Path, date: str, since: str | None, until: str | None,
            name: Path = MY_NAME) -> list[Round]:
    buttons = ContinueButtons.from_directory()
    digits = load_digits()
    my_name = cv2.cvtColor(read_image(name), cv2.COLOR_BGR2GRAY)
    rounds, current = [], Round([])
    for path in sorted(directory.glob(f"{date}_*.png")):
        match = FILE_PATTERN.match(path.stem)
        if not match:
            continue
        clock = match.group(2)
        if since and clock < since or until and clock > until:
            continue
        image = read_image(path)
        found = buttons.find(image)
        if found is None:
            continue
        if found.name in ROUND_END:
            if current.hands:
                rounds.append(current)
                current = Round([])
            continue
        rows = [read_row(image, top, digits, my_name) for top in ROW_TOPS]
        amount, outcome = classify(rows)
        current.hands.append(Hand(path.stem, amount, outcome))
    if current.hands:
        rounds.append(current)
    return rounds


def _rate(count: int, total: int) -> str:
    if not total:
        return "-"
    p = count / total
    margin = 1.96 * math.sqrt(p * (1 - p) / total)
    return f"{p:.0%}（±{margin:.0%}）"


def report(rounds: list[Round]) -> str:
    hands = [hand for rnd in rounds for hand in rnd.hands]
    lines = []
    for index, rnd in enumerate(rounds, 1):
        first, last = rnd.hands[0].stamp[-6:], rnd.hands[-1].stamp[-6:]
        lines.append(f"第 {index} 圈 {first[:2]}:{first[2:4]}–{last[:2]}:{last[2:4]}  "
                     f"{len(rnd.hands)} 局  {rnd.net:+d}")
    total = len(hands)
    counts = {name: sum(1 for h in hands if h.outcome == name)
              for name in ("胡牌", "自摸", "放槍", "被自摸", "沒輸贏", "未知")}
    wins = counts["胡牌"] + counts["自摸"]
    lines += [
        "",
        f"共 {len(rounds)} 圈、{total} 局，總輸贏 {sum(h.my_amount for h in hands):+d}",
        f"胡牌率 {_rate(wins, total)}（胡 {counts['胡牌']}、自摸 {counts['自摸']}）",
        f"放槍率 {_rate(counts['放槍'], total)}",
        f"被自摸 {_rate(counts['被自摸'], total)}、沒輸贏 {_rate(counts['沒輸贏'], total)}",
    ]
    if counts["未知"]:
        lines.append(f"讀不出自己那列：{counts['未知']} 局")
    lines.append("（流局沒有結算畫面，不計入；最後一圈若還沒打完，列出的是目前為止）")
    return "\n".join(lines)


def main() -> None:
    import time

    parser = argparse.ArgumentParser(description="從結算截圖統計實戰戰績")
    parser.add_argument("--date", default=time.strftime("%Y%m%d"), help="YYYYMMDD，預設今天")
    parser.add_argument("--since", help="開始時間 HH:MM")
    parser.add_argument("--until", help="結束時間 HH:MM")
    parser.add_argument("--dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--csv", type=Path, help="逐局結果另存 CSV")
    parser.add_argument("--name", type=Path, default=MY_NAME, help="自己名字的截圖（預設「不沾」）")
    args = parser.parse_args()

    def hhmmss(value):
        return value.replace(":", "").ljust(6, "0") if value else None

    rounds = collect(args.dir, args.date, hhmmss(args.since), hhmmss(args.until), args.name)
    print(report(rounds))
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["圈", "截圖", "輸贏", "結果"])
            for index, rnd in enumerate(rounds, 1):
                for hand in rnd.hands:
                    writer.writerow([index, hand.stamp, hand.my_amount, hand.outcome])


if __name__ == "__main__":
    main()
