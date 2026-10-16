"""大老二提示模式／自動模式（與麻將的 hint/ 分開）。

python -m bigtwo.live                      提示模式：輪到自己時印出建議並顯示浮動小視窗，人來點
python -m bigtwo.live --no-overlay         只印在終端機
python -m bigtwo.live --demo-overlay       預覽小視窗、拖曳調整位置
python -m bigtwo.live --auto               自動點擊：只在對手全是電腦的場使用，啟動時要輸入 y 確認
python -m bigtwo.live --video <mp4> [--start 秒] [--step 0.5]   用錄影回放測試

每一步建議寫進 result/bigtwo_log_<日期>.txt。滑鼠甩到螢幕左上角會中止自動點擊（pyautogui FAILSAFE）。
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Callable, Iterator

import cv2
import numpy as np

from .agents import rule_swap
from .buttons import button_state, center, my_turn, swap_phase
from .cards import HAND_SIZE, RANK_NAMES, SUIT_SYMBOLS, cards_text, rank_of, suit_of
from .combos import Combo, legal_moves
from .counts import read_counts
from .game import View
from .hand_reader import TOP_NORMAL, TOP_RAISED, HandCard, read_hand
from .mc_agent import make_mc_policy
from .rules import DEFAULT_RULES
from .screen import grab_screen, write_image
from .table_reader import read_plays
from .tracker import ME, Tracker

RESULT_DIR = Path("result")
SEAT_NAMES = {0: "自己", 1: "下家", 2: "對家", 3: "上家"}
MOUSE_REST = (450, 60)          # 左上角沙灘（Google Play 版）：游標不要停在牌或按鈕上
CLICK_PAUSE = 0.25
CARD_CLICK = (70, 150)          # 點牌：牌左上角往右下的位置（每張露出 140px 寬；下緣 985 起是牌型提示鈕）
ACTION_COOLDOWN = 0.8
"""點完一輪後至少等這麼久才看下一幀決定下一步：選起的牌要一下子才浮上來，太早再看會以為沒選到、
又點一次反而取消選取（10/16 20:11 實戰換三張被打亂，可能原因之一）。"""
NO_EFFECT_LIMIT = 3
"""連續幾次點了牌、畫面上選起的牌都沒變，就停止自動點擊改成只提示。10/16 電腦版（StarMJ371）
自動模式點擊完全沒反應：遊戲程式有保護（連權杖都不讓查），很可能忽略程式送出的點擊；不做任何繞過。"""
MAX_SHOTS = 300
"""每次執行最多存幾張決定當下的截圖（result/bigtwo/），檢查問題用。"""


def describe(combo: Combo | None) -> str:
    return "PASS" if combo is None else str(combo)


def short(cards) -> str:
    """小視窗用：♥3 ♣4（花色符號＋點數）。"""
    return " ".join(SUIT_SYMBOLS[suit_of(c)] + RANK_NAMES[rank_of(c)] for c in sorted(cards))


def overlay_text(combo: Combo | None) -> str:
    return "PASS" if combo is None else "出 " + short(combo.cards)


class Log:
    def __init__(self, directory: Path = RESULT_DIR, echo: bool = True):
        directory.mkdir(exist_ok=True)
        self.path = directory / f"bigtwo_log_{datetime.now():%Y%m%d}.txt"
        self.echo = echo

    def __call__(self, line: str) -> None:
        if self.echo:
            print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class Clicker:
    """把決定轉成點擊（座標與螢幕擷取相同）。不加隨機延遲或隨機位置。"""

    def __init__(self, click: Callable[[tuple[int, int]], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.click = click or self._pyautogui_click
        self.sleep = sleep

    @staticmethod
    def _pyautogui_click(point):
        import pyautogui

        pyautogui.FAILSAFE = True
        pyautogui.click(*point)
        pyautogui.moveTo(*MOUSE_REST)

    def card(self, hand_card: HandCard) -> None:
        top = TOP_RAISED if hand_card.raised else TOP_NORMAL
        self.click((hand_card.x + CARD_CLICK[0], top + CARD_CLICK[1]))
        self.sleep(CLICK_PAUSE)

    def button(self, name: str) -> None:
        self.click(center(name))
        self.sleep(CLICK_PAUSE)


def screen_frames(interval: float) -> Iterator[tuple[str, np.ndarray]]:
    while True:
        start = time.monotonic()
        yield datetime.now().strftime("%H:%M:%S"), grab_screen()
        time.sleep(max(0.0, interval - (time.monotonic() - start)))


def video_frames(path: Path, step: float, start: float = 0.0) -> Iterator[tuple[str, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    total = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    t = start
    while t < total:
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = capture.read()
        if not ok:
            break
        yield f"{t:7.2f}s", frame
        t += step
    capture.release()


class Advisor:
    def __init__(self, log: Log, clicker: Clicker | None = None, samples: int = 24, seed: int = 0,
                 show: Callable[[str | None], None] = lambda text: None, live: bool = False):
        self.log = log
        self.live = live
        self.show = show
        self.shown: str | None = None
        self.clicker = clicker
        self.policy = make_mc_policy(samples=samples)
        self.rng = Random(seed)
        self.tracker = Tracker()
        self.hand: list[HandCard] | None = None
        self.previous_hand: tuple[int, ...] | None = None
        self.swap_target: frozenset[int] | None = None   # 這局要換的牌（已決定）
        self.swap_finished = False
        self.decided: tuple | None = None
        self.choice: Combo | None = None
        self.turn_frames = 0
        self.busy_until = 0.0
        self.clicked_raised: frozenset[int] | None = None   # 上次點牌前選起的牌
        self.no_effect = 0
        self.shots = 0
        self.shot_dir = RESULT_DIR / "bigtwo"
        self.clock: Callable[[], float] = time.monotonic

    def save_shot(self, frame, label: str) -> None:
        if self.shots >= MAX_SHOTS or self.clicker is None and not self.live:
            return
        self.shots += 1
        write_image(self.shot_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{label}.png", frame)

    def ready(self) -> bool:
        return self.clock() >= self.busy_until

    def acted(self) -> None:
        self.busy_until = self.clock() + ACTION_COOLDOWN

    def clicked_cards(self, hand: list[HandCard], stamp: str, cards) -> None:
        self.clicked_raised = frozenset(c.card for c in hand if c.raised)
        self.log(f"[{stamp}] 點牌 {cards_text(cards)}")
        self.acted()

    def check_effect(self, hand: list[HandCard], stamp: str) -> bool:
        """上次點了牌之後選起的牌有沒有變；連續沒變就停止自動點擊。回傳 False 表示已停止。"""
        if self.clicked_raised is None:
            return True
        raised = frozenset(c.card for c in hand if c.raised)
        if raised != self.clicked_raised:
            self.no_effect = 0
        else:
            self.no_effect += 1
            if self.no_effect >= NO_EFFECT_LIMIT:
                self.log(f"[{stamp}] 點了 {NO_EFFECT_LIMIT} 次牌都沒有選起來：遊戲可能不接受程式送出的點擊，"
                         "自動點擊停止，改為只提示（請自己點）")
                self.clicker = None
                return False
        self.clicked_raised = None
        return True

    def stable_hand(self, frame) -> list[HandCard] | None:
        hand = read_hand(frame)
        cards = None if not hand else tuple(c.card for c in hand)
        steady = cards is not None and cards == self.previous_hand
        self.previous_hand = cards
        return hand if steady else None

    def display(self, text: str | None) -> None:
        if text != self.shown:
            self.shown = text
            self.show(text)

    def step(self, stamp: str, frame: np.ndarray) -> None:
        if swap_phase(frame):
            if self.swap_finished and self.tracker.history:
                self.swap_target, self.swap_finished = None, False   # 新的一局
            if self.swap_target is None:
                self.tracker.reset()
                self.decided = None
            self.handle_swap(stamp, frame)
            return
        hand = read_hand(frame)
        self.hand = hand
        new = self.tracker.my_hand(None if hand is None else [c.card for c in hand])
        new += self.tracker.update(read_plays(frame))
        for play in new:
            self.log(f"[{stamp}] {SEAT_NAMES[play.seat]}出 {describe(play.combo)}")
        # 連續兩幀都是自己的回合、牌桌也沒有還沒確認的牌才下決定（一墩剛結束、別家剛出牌時畫面還在變）
        self.turn_frames = self.turn_frames + 1 if my_turn(frame) else 0
        if self.turn_frames >= 2 and self.tracker.settled:
            self.handle_turn(stamp, frame)
        elif self.turn_frames == 0:
            self.display(None)

    def handle_swap(self, stamp: str, frame) -> None:
        if self.swap_finished:
            return
        hand = self.stable_hand(frame)
        if hand is None or len(hand) != HAND_SIZE:
            return
        if self.swap_target is None:
            cards = tuple(c.card for c in hand)
            give = rule_swap(cards, self.rng)
            self.swap_target = frozenset(give)
            self.log(f"[{stamp}] 換三張：手牌 {cards_text(cards)} → 建議"
                     + (f"換 {cards_text(give)}" if give else "不換"))
            self.display("換三張：" + (short(give) if give else "不換"))
            self.save_shot(frame, "換三張")
        if not self.clicker:
            self.swap_finished = True
            return
        if not self.ready() or not self.check_effect(hand, stamp):
            return
        raised = {c.card for c in hand if c.raised}
        if raised == self.swap_target:
            name = "換牌" if self.swap_target else "不換"
            if button_state(frame, name) == "on":
                self.clicker.button(name)
                self.log(f"[{stamp}] 按「{name}」")
                self.swap_finished = True
                self.acted()
            return
        toggle = [hc for hc in hand if (hc.card in self.swap_target) != hc.raised]
        for hc in toggle:                     # 該選的沒選、不該選的選了：各點一次
            self.clicker.card(hc)
        self.clicked_cards(hand, stamp, [hc.card for hc in toggle])

    def handle_turn(self, stamp: str, frame) -> None:
        hand = self.hand
        if not hand:
            return
        cards = tuple(c.card for c in hand)
        if self.tracker.hand != frozenset(cards):
            return                      # 手牌還沒連續讀到一樣（動畫蓋住幾張等）：等下一幀
        last, last_seat = self.tracker.to_beat()
        key = (cards, last, len(self.tracker.history))
        if key != self.decided:
            counts = read_counts(frame)
            count_tuple = tuple(len(cards) if s == ME else
                                (counts.get(s) if counts.get(s) is not None
                                 else HAND_SIZE - self.tracker.played_by(s)) for s in range(4))
            moves = legal_moves(cards, last, DEFAULT_RULES)
            view = View(ME, cards, count_tuple, last, last_seat, list(self.tracker.history), moves,
                        last is not None, DEFAULT_RULES)
            self.choice = self.policy(view, self.rng) if moves else None
            self.decided = key
            self.clicked_raised = None          # 新的決定：上一手的點牌紀錄不再比較
            target = "帶頭" if last is None else f"壓{SEAT_NAMES[last_seat]}的 {describe(last)}"
            self.log(f"[{stamp}] 輪到自己（{target}；張數 下{count_tuple[1]} 對{count_tuple[2]} 上{count_tuple[3]}）"
                     f"手牌 {cards_text(cards)} → 建議 {describe(self.choice)}")
            self.display(overlay_text(self.choice))
            self.save_shot(frame, "輪到自己")
        if self.clicker and self.ready():
            self.execute(hand, frame, stamp)

    def execute(self, hand: list[HandCard], frame, stamp: str = "") -> None:
        # 先看上次點牌有沒有效果（不論接下來是按出牌、重選或再點牌）：只在點牌前才看時，點成功後
        # 直接按出牌、紀錄沒清掉，下一回合拿上一回合的狀態比較，每回合累加一次，第 3 回合就誤判停止（10/16 22:35）
        if not self.check_effect(hand, stamp):
            return
        choice = self.choice
        if choice is None:
            if button_state(frame, "PASS") == "on":
                self.clicker.button("PASS")
                self.log(f"[{stamp}] 按「PASS」")
                self.acted()
            return
        raised = {c.card for c in hand if c.raised}
        if raised == set(choice.cards):
            if button_state(frame, "出牌") == "on":
                self.clicker.button("出牌")
                self.log(f"[{stamp}] 按「出牌」")
                self.acted()
            return
        if raised - set(choice.cards) and button_state(frame, "重選") == "on":
            self.clicker.button("重選")       # 選了不該選的：清掉，下一輪再選
            self.log(f"[{stamp}] 按「重選」")
            self.acted()
            return
        pick = [hc for hc in hand if hc.card in choice.cards and not hc.raised]
        for hc in pick:
            self.clicker.card(hc)
        self.clicked_cards(hand, stamp, [hc.card for hc in pick])


def confirm_training_room(ask=input) -> bool:
    answer = ask("自動點擊只能在對手全是電腦的場使用。目前的大老二場對手都是電腦嗎？(y/N) ")
    return answer.strip().lower() == "y"


def main() -> None:
    parser = argparse.ArgumentParser(description="大老二提示／自動模式")
    parser.add_argument("--auto", action="store_true", help="自動點擊（只在對手全是電腦的場）")
    parser.add_argument("--video", type=Path, help="用錄影回放測試")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=24, help="蒙地卡羅每個候選模擬幾次")
    parser.add_argument("--no-overlay", action="store_true", help="不顯示浮動小視窗")
    parser.add_argument("--demo-overlay", action="store_true", help="預覽小視窗（可拖曳調整位置）")
    args = parser.parse_args()

    if args.demo_overlay:
        from .overlay import demo
        demo()
        return

    if args.video:
        advisor = Advisor(Log(), None, args.samples)
        for stamp, frame in video_frames(args.video, args.step, args.start):
            advisor.step(stamp, frame)
        return
    clicker = None
    if args.auto:
        if not confirm_training_room():
            print("已取消。真人場請用提示模式（不加 --auto）。")
            return
        clicker = Clicker()
    def worker(show=lambda text: None) -> None:
        advisor = Advisor(Log(), clicker, args.samples, show=show, live=True)
        advisor.log(f"[{datetime.now():%H:%M:%S}] 大老二{'自動' if clicker else '提示'}模式啟動（蒙地卡羅）")
        for stamp, frame in screen_frames(args.interval):
            advisor.step(stamp, frame)

    if args.no_overlay:
        worker()
        return
    from .overlay import Overlay
    window = Overlay()
    window.run(lambda: worker(window.show))


if __name__ == "__main__":
    main()
