"""人機對戰：自己 vs 三家電腦（網路 AI，開實戰護欄），打一圈。

    python -m play [--model models/rl3_0960.pt] [--seed N] [--bot rule]

規則同模擬器：換三張、金牌、按聽、花牌、連莊。打牌時不給提示；每局結束亮四家的牌與台數明細。
操作：點手牌打出；換三張時點選 0～3 張再按「換牌」；能吃碰槓胡時右下出現按鈕；按「聽」後再點要打的牌就是打牌並宣告。
畫面的圖（牌、按鈕、面板）在 play/art.py。
"""
from __future__ import annotations

import argparse
import copy
import queue
import threading
import time
import tkinter as tk
import traceback
from collections import deque
from random import Random

from PIL import Image, ImageTk

from game.rules import Action, ActionType, GameState, Phase
from game.tiles import tile_name
from play import art
from play.game import (HUMAN, MONEY_PER_TAI, SEAT_NAMES, WIND_NAMES, HandSummary, HumanSeat,
                       describe_action, play_round, score_lines, seat_policy)

MODEL = "models/rl3_0960.pt"
WIDTH, HEIGHT = 1400, 900
DISCARD_DELAY = 1.0
"""電腦輪到打牌時，從開始想到打出至少等這麼久（使用者 10/21 要求，2 秒→1 秒）。"""
CLAIM_DELAY = 0.7
"""電腦吃碰槓胡之後停一下，看得清楚。"""
CLAIMS = {ActionType.CHOW, ActionType.PUNG, ActionType.KONG, ActionType.WIN}
DISCARDS = {ActionType.DISCARD, ActionType.DECLARE}
FONT = "Microsoft JhengHei"
HAND_W, HAND_H = 60, 100
RIVER_W, RIVER_H = 32, 52
MELD_W, MELD_H = 32, 52
LAST_COLOR = (230, 50, 50)
DECLARE_COLOR = (40, 120, 240)
BUTTON_COLORS = {
    "win": (215, 55, 55), "claim": (235, 145, 30), "pass": (140, 140, 140),
    "declare": (55, 130, 225), "swap": (35, 160, 105), "continue": (215, 165, 40),
}
COMPASS = (550, 330, 300, 150)
"""中央羅盤 (x, y, 寬, 高)；四家的牌河圍在它四周，各在自己面前。"""
SEATS = {
    # 座位：頭像位置、牌背（起點、每列張數、大小）、副露起點與換行寬度、
    # 牌河（第一列左上角、每列張數、往下或往上長）
    2: dict(avatar=(380, 10), backs=(600, 14, 16, 28, 44), melds=(1090, 14, 1390),
            river=(550, 330 - RIVER_H - 3, 8, "up")),
    3: dict(avatar=(20, 226), backs=(20, 300, 8, 24, 36), melds=(20, 392, 320),
            river=(336, 330, 6, "down")),
    1: dict(avatar=(1080, 226), backs=(1080, 300, 8, 24, 36), melds=(1080, 392, 1390),
            river=(860, 330, 6, "down")),
}
MY_RIVER = (550, 330 + 150 + 8, 8, "down")


def slow_bot(bot, delay: float = DISCARD_DELAY, sleep=time.sleep, clock=time.monotonic):
    """電腦要打牌時等到 delay 秒才打出（思考時間算在內）；吃碰、過不等。"""
    def policy(state, player, rng):
        start = clock()
        action = bot(state, player, rng)
        if action is not None and action.kind in DISCARDS:
            sleep(max(0.0, delay - (clock() - start)))
        return action

    return policy


class App:
    def __init__(self, root: tk.Tk, args) -> None:
        self.root = root
        self.args = args
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, highlightthickness=0, bg="#14502f")
        self.canvas.pack()
        self.photos: dict[int, ImageTk.PhotoImage] = {}
        """PIL 圖 → Tk 圖（art 的圖都有快取，同一張圖只轉一次）。"""
        self.to_gui: queue.Queue = queue.Queue()
        self.from_gui: queue.Queue = queue.Queue()
        self.state: GameState | None = None
        self.actions: list[Action] | None = None
        """等使用者決定的動作；None＝不是自己要決定的時候。"""
        self.selected: list[int] = []
        """換三張選起的手牌位置。"""
        self.declare_mode = False
        self.hover: int | None = None
        """滑鼠指著的手牌位置（浮起來）。"""
        self.messages: deque[str] = deque(maxlen=3)
        self.summary: HandSummary | None = None
        self.round_totals: list[int] | None = None
        self.totals = [0, 0, 0, 0]
        self.hand_number = 1
        self.status = "載入電腦 AI 中…"
        self.hand_tiles: list[int] = []
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Motion>", self.motion)
        threading.Thread(target=self.game_thread, daemon=True).start()
        self.draw()
        self.poll()

    # ---- 牌局執行緒 ----
    def game_thread(self) -> None:
        try:
            bot = self.load_bot()
            human = HumanSeat(self.ask)
            policy = seat_policy(human, slow_bot(bot))
            rng = Random(self.args.seed)
            while True:
                self.to_gui.put(("status", ""))
                totals = play_round(policy, rng, self.hand_end, observer=self.observe)
                self.to_gui.put(("round_end", totals))
                if self.from_gui.get() != "again":
                    return
        except Exception:
            traceback.print_exc()
            self.to_gui.put(("status", "發生錯誤，詳見終端機"))

    def load_bot(self):
        if self.args.bot == "rule":
            from agent.rule_agent import choose_rule_action_for_player
            return choose_rule_action_for_player
        from agent.nn_agent import NetworkPolicy
        return NetworkPolicy.from_file(self.args.model, guard=True)

    def ask(self, state: GameState, actions: list[Action]) -> Action:
        self.to_gui.put(("ask", state, actions))
        return self.from_gui.get()

    def observe(self, state: GameState, player: int, action: Action) -> None:
        self.to_gui.put(("state", copy.deepcopy(state), describe_action(player, action)))
        if player != HUMAN and action.kind in CLAIMS:
            time.sleep(CLAIM_DELAY)

    def hand_end(self, summary: HandSummary) -> None:
        self.to_gui.put(("hand_end", summary))
        self.from_gui.get()

    # ---- 事件 ----
    def poll(self) -> None:
        changed = False
        try:
            while True:
                item = self.to_gui.get_nowait()
                changed = True
                kind = item[0]
                if kind == "status":
                    self.status = item[1]
                elif kind == "state":
                    self.state = item[1]
                    if item[2]:
                        self.messages.append(item[2])
                elif kind == "ask":
                    self.state, self.actions = item[1], item[2]
                    self.selected, self.declare_mode = [], False
                elif kind == "hand_end":
                    self.summary = item[1]
                    self.totals = self.summary.totals
                    self.hand_number = self.summary.number + 1
                    self.messages.clear()
                elif kind == "round_end":
                    self.round_totals = item[1]
        except queue.Empty:
            pass
        if changed:
            self.draw()
        self.root.after(40, self.poll)

    def reply(self, value) -> None:
        self.actions = None
        self.selected, self.declare_mode = [], False
        self.from_gui.put(value)
        self.draw()

    def tag_at(self, x: int, y: int) -> str | None:
        for item in reversed(self.canvas.find_overlapping(x, y, x, y)):
            for tag in self.canvas.gettags(item):
                if tag.startswith(("button:", "tile:")):
                    return tag
        return None

    def click(self, event) -> None:
        tag = self.tag_at(event.x, event.y)
        if tag is None:
            return
        if tag.startswith("button:"):
            self.press(tag[7:])
        else:
            self.tile_clicked(int(tag[5:]))

    def motion(self, event) -> None:
        tag = self.tag_at(event.x, event.y)
        hover = int(tag[5:]) if tag and tag.startswith("tile:") and self.actions else None
        if hover != self.hover:
            self.hover = hover
            self.draw()

    def press(self, name: str) -> None:
        if name == "continue":
            self.summary = None
            self.reply("continue")
            return
        if name in ("again", "quit"):
            self.round_totals = None
            self.totals = [0, 0, 0, 0]
            self.hand_number = 1
            if name == "quit":
                self.root.destroy()
                return
            self.reply("again")
            return
        if self.actions is None:
            return
        if name == "swap":
            self.reply(Action(ActionType.SWAP, tiles=tuple(self.hand_tiles[i] for i in self.selected)))
        elif name == "declare":
            self.declare_mode = not self.declare_mode
            self.draw()
        else:
            self.reply(self.actions[int(name)])

    def tile_clicked(self, index: int) -> None:
        if self.actions is None:
            return
        if self.actions[0].kind == ActionType.SWAP:
            if index in self.selected:
                self.selected.remove(index)
            elif len(self.selected) < 3:
                self.selected.append(index)
            self.draw()
            return
        tile = self.hand_tiles[index]
        kind = ActionType.DECLARE if self.declare_mode else ActionType.DISCARD
        for action in self.actions:
            if action.kind == kind and action.tile == tile:
                self.reply(action)
                return

    # ---- 畫圖工具 ----
    def image(self, x, y, picture: Image.Image, tags=(), anchor="nw") -> None:
        photo = self.photos.get(id(picture))
        if photo is None:
            photo = self.photos[id(picture)] = ImageTk.PhotoImage(picture)
        self.canvas.create_image(x, y, image=photo, anchor=anchor, tags=tags)

    def tile(self, x, y, w, h, tile, tags=(), dim=False, gold=False, outline=None) -> None:
        self.image(x, y, art.tile_image(tile, w, h, dim, gold, outline), tags)

    def back(self, x, y, w, h) -> None:
        self.image(x, y, art.back_image(w, h))

    def text(self, x, y, text, size=13, color="white", anchor="nw", bold=False, width=0, shadow=True):
        weight = "bold" if bold else "normal"
        if shadow:
            self.canvas.create_text(x + 1, y + 1, text=text, fill="#06140d", anchor=anchor, width=width,
                                    font=(FONT, size, weight))
        self.canvas.create_text(x, y, text=text, fill=color, anchor=anchor, width=width, font=(FONT, size, weight))

    def button(self, x, y, label, name, color="claim", w=None, h=46) -> int:
        w = w or 40 + 20 * len(label)
        tags = (f"button:{name}",)
        self.image(x, y, art.button_image(w, h, BUTTON_COLORS[color]), tags)
        self.canvas.create_text(x + w / 2, y + h / 2, text=label, fill="white", font=(FONT, 16, "bold"), tags=tags)
        return w

    # ---- 畫面 ----
    def draw(self) -> None:
        self.canvas.delete("all")
        self.image(0, 0, art.felt_image(WIDTH, HEIGHT))
        state = self.state
        if state is None:
            self.text(WIDTH / 2, HEIGHT / 2, self.status or "準備中…", 24, anchor="center", bold=True)
            return
        self.draw_center(state)
        for seat in (1, 2, 3):
            self.draw_opponent(state, seat)
        self.draw_me(state)
        if self.summary is not None:
            self.draw_summary(self.summary)
        if self.round_totals is not None:
            self.draw_round_end(self.round_totals)

    def active(self, state: GameState, seat: int) -> bool:
        if state.phase == Phase.RESPONSE:
            return False
        return state.current_player == seat and state.phase in (Phase.DRAW, Phase.DISCARD, Phase.SWAP)

    def plate(self, state: GameState, seat: int, x: int, y: int, size: int = 64) -> None:
        """頭像（門風字，輪到的人金框）、名字、輸贏、莊與聽的章。"""
        ring = (255, 205, 60) if self.active(state, seat) else (235, 235, 235)
        self.image(x, y, art.avatar_image(size, WIND_NAMES[state.seat_winds[seat]], ring))
        tx = x + size + 10
        self.text(tx, y + 4, SEAT_NAMES[seat], 15, bold=True,
                  color="#ffd84d" if self.active(state, seat) else "white")
        total = self.totals[seat]
        self.text(tx, y + 32, f"{total:+d} 台（{total * MONEY_PER_TAI:+d}）", 12,
                  color="#bff5d3" if total >= 0 else "#ffb3b3")
        bx = tx + 64
        if state.dealer == seat:
            self.image(bx, y + 2, art.badge_image("莊", (205, 40, 40)))
            if state.dealer_streak:
                self.text(bx + 34, y + 6, f"連{state.dealer_streak}", 12, color="#ffd0d0", bold=True)
            bx += 66
        if state.declared[seat]:
            self.image(bx, y + 2, art.badge_image("聽", (40, 110, 220)))

    def river(self, x, y, per_row, direction, tiles, gold, highlight_last=False) -> None:
        step = RIVER_H + 3 if direction == "down" else -(RIVER_H + 3)
        for i, tile in enumerate(tiles):
            row, col = divmod(i, per_row)
            last = highlight_last and i == len(tiles) - 1
            tx, ty = x + col * (RIVER_W + 2), y + row * step
            self.tile(tx, ty, RIVER_W, RIVER_H, tile, gold=tile in gold,
                      outline=LAST_COLOR if last else None)
            if last:
                self.canvas.create_polygon(tx + RIVER_W / 2 - 7, ty - 10, tx + RIVER_W / 2 + 7, ty - 10,
                                           tx + RIVER_W / 2, ty - 1, fill="#ff4040", outline="white")

    def melds(self, x, y, melds, w, h, right: int, show_concealed: bool, gold=()) -> tuple[int, int]:
        """副露一組一組排，超過 right 換行；回傳下一個位置。"""
        x0 = x
        for meld in melds:
            width = len(meld.tiles) * (w + 1) + 10
            if x + width > right and x > x0:
                x, y = x0, y + h + 8
            concealed = meld.kind == "kong" and meld.from_player is None
            for j, tile in enumerate(meld.tiles):
                if concealed and not show_concealed and j in (0, 3):
                    self.back(x, y, w, h)
                else:
                    self.tile(x, y, w, h, tile, gold=tile in gold)
                x += w + 1
            x += 10
        return x, y

    def flowers(self, x, y, flowers, w, h) -> None:
        for flower in flowers:
            self.tile(x, y, w, h, flower)
            x += w + 1

    def draw_center(self, state: GameState) -> None:
        self.draw_info(state)
        self.draw_compass(state)

    def draw_info(self, state: GameState) -> None:
        """左上角：局數、莊家、牌牆、金牌、最近的動作。"""
        x, y = 16, 10
        self.image(x, y, art.panel_image(340, 200))
        dealer = SEAT_NAMES[state.dealer] + (f"（連 {state.dealer_streak}）" if state.dealer_streak else "")
        self.text(x + 18, y + 12, f"東風圈　第 {self.hand_number} 局", 16, bold=True, color="#ffe08a")
        self.text(x + 18, y + 42, f"莊家 {dealer}　·　牌牆剩 {state.drawable} 張", 11, color="#d8efe2")
        self.text(x + 18, y + 80, "金牌", 13, bold=True, color="#ffd84d")
        for i, tile in enumerate(state.gold_tiles):
            self.tile(x + 66 + i * 34, y + 66, 32, 50, tile, gold=True)
        for i, message in enumerate(self.messages):
            self.text(x + 18, y + 128 + i * 20, message, 12, color="#cfe9da")

    def draw_compass(self, state: GameState) -> None:
        """中央：四邊是各家門風（輪到的人亮起），中間是牌牆剩幾張或剛打出的牌。"""
        x, y, w, h = COMPASS
        self.image(x, y, art.panel_image(w, h, alpha=150))
        cx, cy = x + w / 2, y + h / 2
        spots = {HUMAN: (cx, y + h - 8, "s"), 1: (x + w - 12, cy, "e"), 2: (cx, y + 8, "n"), 3: (x + 12, cy, "w")}
        for seat, (sx, sy, anchor) in spots.items():
            on = self.active(state, seat) or (state.phase == Phase.RESPONSE and state.discard_player == seat)
            self.text(sx, sy, WIND_NAMES[state.seat_winds[seat]], 22 if on else 15, anchor=anchor, bold=on,
                      color="#ffd84d" if on else "#9fbfae")
        if state.last_discard is not None and state.phase == Phase.RESPONSE:
            self.tile(cx - 22, cy - 38, 44, 72, state.last_discard, outline=LAST_COLOR)
        else:
            self.text(cx, cy - 12, f"{state.drawable}", 26, anchor="center", bold=True, color="#e8f5ec")
            self.text(cx, cy + 20, "牌牆剩餘", 10, anchor="center", color="#9fbfae")

    def prompt(self, state: GameState) -> str:
        if not self.actions:
            if state.declared[HUMAN]:
                return "已聽牌：遊戲代打中"
            return ""
        if self.actions[0].kind == ActionType.SWAP:
            return "換三張：點 0～3 張要換掉的牌，再按右下按鈕"
        if state.phase == Phase.RESPONSE:
            return f"{SEAT_NAMES[state.discard_player]}打出 {tile_name(state.last_discard)}：要吃碰嗎？"
        if self.declare_mode:
            return "聽牌：點藍框的牌打出並宣告"
        return "輪到你：點一張牌打出"

    def draw_opponent(self, state: GameState, seat: int) -> None:
        layout = SEATS[seat]
        player = state.players[seat]
        gold = set(state.gold_tiles)
        self.plate(state, seat, *layout["avatar"])
        bx, by, per_row, bw, bh = layout["backs"]
        for i in range(sum(player.hand)):
            row, col = divmod(i, per_row)
            self.back(bx + col * (bw + 2), by + row * (bh + 4), bw, bh)
        mx, my, right = layout["melds"]
        _, y = self.melds(mx, my, player.melds, MELD_W, MELD_H, right, show_concealed=False, gold=gold)
        # 花牌另起一行（接在副露後面會伸進牌河）
        self.flowers(mx, y + MELD_H + 8 if player.melds else my, player.flowers, MELD_W, MELD_H)
        last = state.phase == Phase.RESPONSE and state.discard_player == seat
        self.river(*layout["river"], player.discards, gold, highlight_last=last)

    def draw_me(self, state: GameState) -> None:
        me = state.players[HUMAN]
        gold = set(state.gold_tiles)
        last = state.phase == Phase.RESPONSE and state.discard_player == HUMAN
        self.river(*MY_RIVER, me.discards, gold, highlight_last=last)
        self.plate(state, HUMAN, 20, 664)
        x, y = self.melds(300, 664, me.melds, 40, 66, 900, show_concealed=True, gold=gold)
        self.flowers(x, y, me.flowers, 40, 66)
        # 手牌：排序，剛摸進的那張放最右邊
        tiles = [t for t in range(34) for _ in range(me.hand[t])]
        drawn = None
        if state.phase == Phase.DISCARD and state.current_player == HUMAN and not state.claimed \
                and state.last_drawn is not None and state.last_drawn in tiles:
            tiles.remove(state.last_drawn)
            drawn = state.last_drawn
        self.hand_tiles = tiles + ([drawn] if drawn is not None else [])
        actions = self.actions or []
        swapping = bool(actions) and actions[0].kind == ActionType.SWAP
        playable = {a.tile for a in actions
                    if a.kind == (ActionType.DECLARE if self.declare_mode else ActionType.DISCARD)}
        choosing = bool(actions) and (swapping or state.phase == Phase.DISCARD)
        x0 = 150
        for i, tile in enumerate(self.hand_tiles):
            x = x0 + i * (HAND_W + 2) + (18 if drawn is not None and i == len(self.hand_tiles) - 1 else 0)
            y = 780
            if i in self.selected:
                y -= 26
            elif choosing and i == self.hover and (swapping or tile in playable):
                y -= 12
            dim = choosing and not swapping and tile not in playable
            outline = DECLARE_COLOR if self.declare_mode and tile in playable else \
                (35, 160, 105) if i in self.selected else None
            self.tile(x, y, HAND_W, HAND_H, tile, tags=(f"tile:{i}",), dim=dim, gold=tile in gold,
                      outline=outline)
        prompt = self.prompt(state)
        if prompt:
            self.text(x0, 750, prompt, 13, bold=True, color="#ffe08a", anchor="w")
        self.draw_buttons(actions, swapping)

    def draw_buttons(self, actions: list[Action], swapping: bool) -> None:
        if not actions:
            return
        buttons: list[tuple[str, str, str]] = []
        if swapping:
            label = f"換牌（{len(self.selected)} 張）" if self.selected else "不換"
            buttons.append((label, "swap", "swap"))
        else:
            has_declare = False
            for index, action in enumerate(actions):
                kind = action.kind
                if kind == ActionType.WIN:
                    buttons.append(("自摸" if self.state.phase == Phase.DISCARD else "胡", str(index), "win"))
                elif kind == ActionType.PUNG:
                    buttons.append(("碰", str(index), "claim"))
                elif kind == ActionType.KONG:
                    buttons.append((f"槓 {tile_name(action.tile)}", str(index), "claim"))
                elif kind == ActionType.CHOW:
                    buttons.append(("吃 " + "".join(tile_name(t)[0] for t in action.tiles)
                                    + tile_name(action.tiles[0])[1], str(index), "claim"))
                elif kind == ActionType.PASS:
                    buttons.append(("過", str(index), "pass"))
                elif kind == ActionType.DECLARE:
                    has_declare = True
            if has_declare:
                buttons.append(("取消聽" if self.declare_mode else "聽", "declare", "declare"))
        # 靠右排在手牌上方
        x = WIDTH - 24
        for label, name, color in reversed(buttons):
            w = 40 + 20 * len(label)
            x -= w
            self.button(x, 704, label, name, color, w=w)
            x -= 12

    def draw_summary(self, summary: HandSummary) -> None:
        result = summary.result
        state = result.state
        self.image(0, 0, art.panel_image(WIDTH, HEIGHT, alpha=120))
        px, py = 110, 40
        self.image(px, py, art.panel_image(1180, 820, alpha=235, border=(230, 190, 80)))
        dealer = SEAT_NAMES[summary.dealer] + (f"（連 {summary.streak}）" if summary.streak else "")
        self.text(px + 36, py + 24, f"第 {summary.number} 局　莊家：{dealer}", 16, color="#d8efe2")
        if result.winner is None:
            outcome = "流局"
        elif result.win_by_discard:
            outcome = f"{SEAT_NAMES[result.winner]} 胡牌　{SEAT_NAMES[result.discarder]} 放槍"
        else:
            outcome = f"{SEAT_NAMES[result.winner]} 自摸"
        if result.score is not None:
            outcome += f"　{result.score.tai} 台"
        self.text(px + 36, py + 54, outcome, 26, color="#ffd84d", bold=True)
        self.text(px + 36, py + 100, "　".join(score_lines(result)), 13, color="#ffe9a8", width=1100)
        gold = set(state.gold_tiles)
        for seat in range(4):
            y = py + 150 + seat * 150
            player = state.players[seat]
            winner = seat == result.winner
            if winner:
                self.canvas.create_rectangle(px + 20, y - 10, px + 1160, y + 130, outline="#e6be50", width=2)
            ring = (255, 205, 60) if winner else (235, 235, 235)
            self.image(px + 36, y, art.avatar_image(56, WIND_NAMES[state.seat_winds[seat]], ring))
            pay = summary.pay[seat]
            color = "#bff5d3" if pay > 0 else "#ffb3b3" if pay < 0 else "white"
            self.text(px + 104, y + 2, SEAT_NAMES[seat], 15, bold=True, color="#ffd84d" if winner else "white")
            self.text(px + 104, y + 30, f"這局 {pay:+d} 台（{pay * MONEY_PER_TAI:+d}）"
                                        f"　合計 {summary.totals[seat]:+d} 台", 12, color=color)
            if state.declared[seat]:
                self.image(px + 170, y, art.badge_image("聽", (40, 110, 220), 26))
            x, ty = self.melds(px + 104, y + 58, player.melds, 34, 56, px + 1150, show_concealed=True, gold=gold)
            tiles = [t for t in range(34) for _ in range(player.hand[t])]
            win_tile = result.win_tile if winner else None
            if win_tile is not None and win_tile in tiles:
                tiles.remove(win_tile)
                tiles.append(win_tile)
            for i, tile in enumerate(tiles):
                is_win = win_tile is not None and i == len(tiles) - 1
                self.tile(x + (10 if is_win else 0), ty, 34, 56, tile, gold=tile in gold,
                          outline=LAST_COLOR if is_win else None)
                x += 35
            self.flowers(x + 24, ty, player.flowers, 34, 56)
        self.button(px + 1180 - 180, py + 820 - 70, "繼續", "continue", "continue", w=150)

    def draw_round_end(self, totals: list[int]) -> None:
        self.image(0, 0, art.panel_image(WIDTH, HEIGHT, alpha=150))
        px, py, pw, ph = 400, 210, 600, 460
        self.image(px, py, art.panel_image(pw, ph, alpha=240, border=(230, 190, 80)))
        self.text(px + pw / 2, py + 30, "一圈結束", 28, anchor="n", bold=True, color="#ffd84d")
        order = sorted(range(4), key=lambda seat: -totals[seat])
        for rank, seat in enumerate(order):
            y = py + 100 + rank * 64
            color = "#ffd84d" if seat == HUMAN else "white"
            self.text(px + 70, y, f"{rank + 1}", 24, bold=True, color=color)
            self.text(px + 120, y + 4, SEAT_NAMES[seat], 20, bold=True, color=color)
            amount = totals[seat]
            self.text(px + pw - 60, y + 6, f"{amount:+d} 台（{amount * MONEY_PER_TAI:+d}）", 18, anchor="ne",
                      color="#bff5d3" if amount >= 0 else "#ffb3b3")
        self.button(px + 110, py + ph - 84, "再一圈", "again", "swap", w=170)
        self.button(px + pw - 280, py + ph - 84, "結束", "quit", "pass", w=170)


def main() -> None:
    parser = argparse.ArgumentParser(description="自己 vs 三家電腦，打一圈")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--bot", choices=("nn", "rule"), default="nn")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = tk.Tk()
    root.title("麻將：自己 vs 三家電腦")
    root.resizable(False, False)
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
