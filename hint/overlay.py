"""浮在遊戲畫面上的建議小視窗（單螢幕用）。

視窗設為「螢幕擷取時看不到」（Windows 10 2004 以上的 WDA_EXCLUDEFROMCAPTURE），
人看得到，但程式擷取畫面時它不存在，不會擋到辨識。
"""
from __future__ import annotations

import ctypes
import queue
import threading
import tkinter as tk
from typing import Callable

WDA_EXCLUDEFROMCAPTURE = 0x11
GEOMETRY = "560x70+680+150"
"""放在對家牌牆上方，不擋自己的手牌、按鈕與可吃碰的放大牌。"""


class Overlay:
    def __init__(self) -> None:
        self.messages: queue.Queue[str] = queue.Queue()
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.9)
        self.root.geometry(GEOMETRY)
        self.label = tk.Label(self.root, text="提示模式啟動", font=("Microsoft JhengHei", 24, "bold"),
                              fg="#ffffff", bg="#202020")
        self.label.pack(fill=tk.BOTH, expand=True)
        self.root.update()
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        if not ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            print("警告：無法把視窗排除在螢幕擷取之外，小視窗可能擋到辨識", flush=True)

    def show(self, text: str) -> None:
        """可從其他執行緒呼叫。"""
        self.messages.put(text)

    def _poll(self) -> None:
        try:
            while True:
                self.label.config(text=self.messages.get_nowait())
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def run(self, worker: Callable[[], None]) -> None:
        """在背景執行 worker（擷取與辨識），主執行緒跑視窗；worker 結束或 Ctrl+C 時關閉。"""
        def target() -> None:
            try:
                worker()
            finally:
                self.root.after(0, self.root.destroy)

        threading.Thread(target=target, daemon=True).start()
        self._poll()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
