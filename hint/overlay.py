"""浮在遊戲畫面上的建議小視窗（單螢幕用）。

- 視窗設為「螢幕擷取時看不到」（Windows 10 2004 以上的 WDA_EXCLUDEFROMCAPTURE），
  人看得到，但程式擷取畫面時它不存在，不會擋到辨識。
- 依建議種類上色：胡紅色、吃碰槓綠色、聽紫色、換三張橘色、打牌白色。
- 沒有需要決定的事時變成半透明的「等待中」，避免看到過期的建議。
- 左鍵拖曳可移動，位置記在 hint/overlay_position.txt；右鍵關閉。
"""
from __future__ import annotations

import ctypes
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Callable

WDA_EXCLUDEFROMCAPTURE = 0x11
WIDTH, HEIGHT = 620, 86
DEFAULT_POSITION = (650, 140)
"""預設在對家牌牆上方，不擋自己的手牌、按鈕與可吃碰的放大牌。"""
POSITION_FILE = Path(__file__).with_name("overlay_position.txt")
TABLE_POSITION_FILE = Path(__file__).with_name("danger_position.txt")
TABLE_DEFAULT = (20, 440)
"""放槍表預設在左邊（上家牌牆外側），不擋手牌、牌河與按鈕。"""
TABLE_WIDTH, TABLE_HEIGHT = 300, 190
TABLE_IDLE = "放槍機率：等輪到自己"
MELD_DEFAULTS = {"left": (10, 150), "top": (1170, 20), "right": (1610, 150)}
"""三家副露小標籤的預設位置（使用者 10/22 要求「在對手旁邊顯示他吃碰槓什麼牌」）：上家、下家在各自頭像上方，
對家在名牌右邊。位置存 hint/meld_position_<座位>.txt。"""
MELD_COLOR = "#ffd54f"

IDLE_TEXT = "等待中…"
ACTIVE_ALPHA = 0.92
IDLE_ALPHA = 0.45
BACKGROUND = "#1b1b1b"

COLORS = (
    ("胡", "#ff5252"),
    ("聽", "#d17bff"),
    ("換三張", "#ffab40"),
    ("碰", "#69f0ae"),
    ("吃", "#69f0ae"),
    ("槓", "#69f0ae"),
    ("不要", "#bdbdbd"),
)


def color_for(text: str) -> str:
    for keyword, color in COLORS:
        if keyword in text:
            return color
    return "#ffffff"


def load_position(path: Path = POSITION_FILE,
                  default: tuple[int, int] = DEFAULT_POSITION) -> tuple[int, int]:
    try:
        x, y = (int(value) for value in path.read_text(encoding="utf-8").split(","))
        return x, y
    except (OSError, ValueError):
        return default


def save_position(x: int, y: int, path: Path = POSITION_FILE) -> None:
    try:
        path.write_text(f"{x},{y}", encoding="utf-8")
    except OSError:
        pass


class DangerTable:
    """放槍表小視窗：輪到自己時手上放槍機率最高的幾張（hint.advisor.danger_table）。
    左鍵拖曳移動（位置存 hint/danger_position.txt）、右鍵隱藏；擷取畫面看不到它。"""

    def __init__(self, root: tk.Tk) -> None:
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", ACTIVE_ALPHA)
        x, y = load_position(TABLE_POSITION_FILE, TABLE_DEFAULT)
        self.window.geometry(f"{TABLE_WIDTH}x{TABLE_HEIGHT}+{x}+{y}")
        self.window.configure(bg=BACKGROUND)
        self.label = tk.Label(self.window, font=("Microsoft JhengHei", 15, "bold"), fg="#e0e0e0",
                              bg=BACKGROUND, justify=tk.LEFT, anchor="nw")
        self.label.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.set(None)
        self._drag_from: tuple[int, int] | None = None
        for widget in (self.window, self.label):
            widget.bind("<ButtonPress-1>", self._start)
            widget.bind("<B1-Motion>", self._drag)
            widget.bind("<ButtonRelease-1>", self._end)
            widget.bind("<Button-3>", lambda _: self.window.withdraw())
        self.window.update()
        hwnd = ctypes.windll.user32.GetParent(self.window.winfo_id())
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)

    def set(self, text: str | None) -> None:
        self.label.config(text=text or TABLE_IDLE, fg="#e0e0e0" if text else "#9e9e9e")
        self.window.attributes("-alpha", ACTIVE_ALPHA if text else IDLE_ALPHA)

    def _start(self, event: tk.Event) -> None:
        self._drag_from = (event.x_root - self.window.winfo_x(),
                           event.y_root - self.window.winfo_y())

    def _drag(self, event: tk.Event) -> None:
        if self._drag_from is not None:
            dx, dy = self._drag_from
            self.window.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _end(self, _: tk.Event) -> None:
        self._drag_from = None
        save_position(self.window.winfo_x(), self.window.winfo_y(), TABLE_POSITION_FILE)


class MeldLabel:
    """對手旁邊的副露小標籤：沒有副露時隱藏；左鍵拖曳（位置存檔）、右鍵隱藏到下次有變化；擷取畫面看不到它。"""

    def __init__(self, root: tk.Tk, seat: str) -> None:
        self.path = Path(__file__).with_name(f"meld_position_{seat}.txt")
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", ACTIVE_ALPHA)
        self.window.configure(bg=BACKGROUND)
        x, y = load_position(self.path, MELD_DEFAULTS[seat])
        self.window.geometry(f"+{x}+{y}")
        self.label = tk.Label(self.window, font=("Microsoft JhengHei", 15, "bold"), fg=MELD_COLOR,
                              bg=BACKGROUND, padx=10, pady=4)
        self.label.pack()
        self._drag_from: tuple[int, int] | None = None
        for widget in (self.window, self.label):
            widget.bind("<ButtonPress-1>", self._start)
            widget.bind("<B1-Motion>", self._drag)
            widget.bind("<ButtonRelease-1>", self._end)
            widget.bind("<Button-3>", lambda _: self.window.withdraw())
        self.window.update()
        hwnd = ctypes.windll.user32.GetParent(self.window.winfo_id())
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        self.window.withdraw()

    def set(self, text: str | None) -> None:
        if not text:
            self.window.withdraw()
            return
        self.label.config(text=text)
        self.window.deiconify()
        self.window.attributes("-topmost", True)

    def _start(self, event: tk.Event) -> None:
        self._drag_from = (event.x_root - self.window.winfo_x(), event.y_root - self.window.winfo_y())

    def _drag(self, event: tk.Event) -> None:
        if self._drag_from is not None:
            dx, dy = self._drag_from
            self.window.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _end(self, _: tk.Event) -> None:
        self._drag_from = None
        save_position(self.window.winfo_x(), self.window.winfo_y(), self.path)


class Overlay:
    def __init__(self) -> None:
        self.messages: queue.Queue[str | None] = queue.Queue()
        self.tables: queue.Queue[str | None] = queue.Queue()
        self.meld_texts: queue.Queue[dict[str, str | None]] = queue.Queue()
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        x, y = load_position()
        self.root.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")
        self.root.configure(bg=BACKGROUND)
        self.label = tk.Label(self.root, font=("Microsoft JhengHei", 28, "bold"),
                              bg=BACKGROUND, wraplength=WIDTH - 20)
        self.label.pack(fill=tk.BOTH, expand=True)
        self._set(None)
        self._drag_from: tuple[int, int] | None = None
        for widget in (self.root, self.label):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._drag)
            widget.bind("<ButtonRelease-1>", self._end_drag)
            widget.bind("<Button-3>", lambda _: self.root.destroy())
        self.root.update()
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        if not ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            print("警告：無法把視窗排除在螢幕擷取之外，小視窗可能擋到辨識", flush=True)
        self.table = DangerTable(self.root)
        self.meld_labels = {seat: MeldLabel(self.root, seat) for seat in MELD_DEFAULTS}

    def show(self, text: str | None) -> None:
        """顯示建議；None 代表目前不需要決定。可從其他執行緒呼叫。"""
        self.messages.put(text)

    def show_table(self, text: str | None) -> None:
        """更新放槍表；None 代表還沒輪到自己。可從其他執行緒呼叫。"""
        self.tables.put(text)

    def show_melds(self, texts: dict[str, str | None]) -> None:
        """更新三家的副露小標籤（座位 left/top/right → 文字，None 隱藏）。可從其他執行緒呼叫。"""
        self.meld_texts.put(texts)

    def _set(self, text: str | None) -> None:
        if text is None:
            self.label.config(text=IDLE_TEXT, fg="#9e9e9e")
            self.root.attributes("-alpha", IDLE_ALPHA)
        else:
            self.label.config(text=text, fg=color_for(text))
            self.root.attributes("-alpha", ACTIVE_ALPHA)

    def _start_drag(self, event: tk.Event) -> None:
        self._drag_from = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _drag(self, event: tk.Event) -> None:
        if self._drag_from is not None:
            dx, dy = self._drag_from
            self.root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _end_drag(self, _: tk.Event) -> None:
        self._drag_from = None
        save_position(self.root.winfo_x(), self.root.winfo_y())

    def _poll(self) -> None:
        latest = ...
        try:
            while True:
                latest = self.messages.get_nowait()
        except queue.Empty:
            pass
        if latest is not ...:
            self._set(latest)
        table = ...
        try:
            while True:
                table = self.tables.get_nowait()
        except queue.Empty:
            pass
        if table is not ...:
            self.table.set(table)
        texts = None
        try:
            while True:
                texts = self.meld_texts.get_nowait()
        except queue.Empty:
            pass
        if texts is not None:
            for seat, label in self.meld_labels.items():
                label.set(texts.get(seat))
        self.root.after(50, self._poll)

    def run(self, worker: Callable[[], None]) -> None:
        """在背景執行 worker（擷取與辨識），主執行緒跑視窗；worker 結束、右鍵或 Ctrl+C 時關閉。"""
        def target() -> None:
            try:
                worker()
            finally:
                try:
                    self.root.after(0, self.root.destroy)
                except (RuntimeError, tk.TclError):
                    pass  # 視窗已被右鍵關閉

        threading.Thread(target=target, daemon=True).start()
        self._poll()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


def demo(seconds: float = 2.0) -> None:
    """依序顯示幾種建議，用來檢查外觀與位置。"""
    overlay = Overlay()

    def worker() -> None:
        overlay.show_table("\n".join(("放槍機率（手上最高 5 張）", "五條   8.3%  上家★",
                                      "三萬   6.1%  上家★", "七筒   4.0%  對家",
                                      "東     0.8%  下家", "北     0.0%")))
        overlay.show_melds({"left": "上家　碰 白　吃 七八九筒", "top": "對家　碰 六萬",
                            "right": "下家　碰 東　槓 二筒"})
        for text in ("打 九筒", "碰 二萬", "打 四筒，並按「聽」", "胡！", None):
            overlay.show(text)
            time.sleep(seconds)

    overlay.run(worker)
