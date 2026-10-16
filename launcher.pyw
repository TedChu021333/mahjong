"""按鈕啟動器：桌面捷徑「麻將大老二 AI」雙擊打開。

按鈕清單沿用 start.py 的 MENU（加功能只要改那裡）。每個按鈕另開一個命令列視窗執行
`python start.py <編號>`，程式結束後視窗留著（看得到記錄與錯誤），要停止就關掉那個視窗。
自動模式一樣會在那個視窗問「對手是不是全是電腦」，要輸入 y 才開始。
"""
from __future__ import annotations

import os
import subprocess
import tkinter as tk
from pathlib import Path

import start

CREATE_NEW_CONSOLE = 0x00000010


def command(number: int) -> list[str]:
    """第 number 項（1 起）的指令：新命令列視窗執行完不關閉。"""
    return ["cmd", "/k", start.python(), str(start.ROOT / "start.py"), str(number)]


def launch(number: int) -> None:
    subprocess.Popen(command(number), cwd=start.ROOT, creationflags=CREATE_NEW_CONSOLE,
                     env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def main() -> None:
    root = tk.Tk()
    root.title("麻將大老二 AI")
    root.resizable(False, False)
    tk.Label(root, text="選一個功能（會另開視窗執行）", font=("Microsoft JhengHei", 12)).pack(padx=16, pady=(12, 6))
    for number, (name, _) in enumerate(start.MENU, 1):
        auto = "自動" in name
        tk.Button(root, text=name, width=40, font=("Microsoft JhengHei", 12),
                  bg="#ffe0b2" if auto else "#e3f2fd",
                  command=lambda n=number: launch(n)).pack(padx=16, pady=3)
    tk.Label(root, text="自動模式只在對手全是電腦的場使用", fg="#b71c1c",
             font=("Microsoft JhengHei", 10)).pack(padx=16, pady=(6, 12))
    root.mainloop()


if __name__ == "__main__":
    main()
