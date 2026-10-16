"""大老二：浮在遊戲畫面上的建議小視窗（從麻將的 hint/overlay.py 複製，兩邊分開維護）。

- 視窗設為「螢幕擷取時看不到」（Windows 10 2004 以上的 WDA_EXCLUDEFROMCAPTURE），
  人看得到，但程式擷取畫面時它不存在，不會擋到辨識。
- 依建議種類上色：出牌白色、PASS 灰色、換三張橘色。
- 沒有需要決定的事時變成半透明的「等待中」，避免看到過期的建議。
- 左鍵拖曳可移動，位置記在 bigtwo/overlay_position.txt；右鍵關閉。
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
DEFAULT_POSITION = (140, 95)
"""預設在左上角沙灘（「代打」右邊）：不擋手牌、按鈕、四家出牌區與剩幾張的數字。"""
POSITION_FILE = Path(__file__).with_name("overlay_position.txt")

IDLE_TEXT = "等待中…"
ACTIVE_ALPHA = 0.92
IDLE_ALPHA = 0.45
BACKGROUND = "#1b1b1b"

COLORS = (
    ("換三張", "#ffab40"),
    ("PASS", "#bdbdbd"),
)


def color_for(text: str) -> str:
    for keyword, color in COLORS:
        if keyword in text:
            return color
    return "#ffffff"


def load_position(path: Path = POSITION_FILE) -> tuple[int, int]:
    try:
        x, y = (int(value) for value in path.read_text(encoding="utf-8").split(","))
        return x, y
    except (OSError, ValueError):
        return DEFAULT_POSITION


def save_position(x: int, y: int, path: Path = POSITION_FILE) -> None:
    try:
        path.write_text(f"{x},{y}", encoding="utf-8")
    except OSError:
        pass


class Overlay:
    def __init__(self) -> None:
        self.messages: queue.Queue[str | None] = queue.Queue()
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

    def show(self, text: str | None) -> None:
        """顯示建議；None 代表目前不需要決定。可從其他執行緒呼叫。"""
        self.messages.put(text)

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
        for text in ("換三張：♥5 ♦7 ♦8", "出 ♣2 ♠2", "出 ♥3 ♣4 ♦5 ♠6 ♣2", "PASS", None):
            overlay.show(text)
            time.sleep(seconds)

    overlay.run(worker)
