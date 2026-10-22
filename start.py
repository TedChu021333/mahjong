"""所有執行指令的總選單。

python start.py       顯示選單，輸入數字啟動（任何終端機都可以，會自動用 .venv-train）
python start.py 4     直接啟動第 4 項（後面可再接參數，例如 python start.py 1 --efficiency-turns 6）
VS Code：Ctrl+Shift+P →「Tasks: Run Task」→「總選單」
這是唯一的啟動入口（10/17 刪掉 run.sh、run_hint.sh、start.sh 等腳本）。

自動模式只在對手全是電腦的場使用，啟動後各程式還會再問一次 y 確認。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "models/rl3_0960.pt"

MENU = [
    ("麻將：自動模式（訓練場）", ["-m", "hint", "--auto", "--ai", "nn", "--model", MODEL,
                                     "--efficiency-turns", "6"]),
    ("麻將：自動模式＋極致進攻（訓練場）", ["-m", "hint", "--auto", "--ai", "nn", "--model", MODEL,
                                   "--attack"]),
    ("麻將：提示模式（真人場，不自動點擊）", ["-m", "hint", "--ai", "nn", "--model", MODEL]),
    ("大老二：自動模式（只限對手全是電腦的場）", ["-m", "bigtwo.live", "--auto"]),
    ("大老二：提示模式（不自動點擊）", ["-m", "bigtwo.live"]),
    ("麻將：今天戰績", ["-m", "eval.stats"]),
    ("麻將：自己 vs 三家電腦（一圈）", ["-m", "play", "--model", MODEL]),
]


def python() -> str:
    venv = ROOT / ".venv-train" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def choose(argv: list[str], ask=input) -> tuple[int | None, list[str]]:
    """回傳 (選項編號 0 起, 額外參數)；None 表示離開。"""
    if argv and argv[0].isdigit():
        number, extra = int(argv[0]), argv[1:]
    else:
        print("\n".join(f"  {i}. {name}" for i, (name, _) in enumerate(MENU, 1)))
        print("  0. 離開")
        answer = ask("選擇：").strip()
        if not answer.isdigit():
            return None, []
        number, extra = int(answer), argv
    if not 1 <= number <= len(MENU):
        return None, []
    return number - 1, extra


def main(argv: list[str] | None = None) -> int:
    index, extra = choose(sys.argv[1:] if argv is None else argv)
    if index is None:
        return 0
    name, args = MENU[index]
    print(f"啟動：{name}", flush=True)
    return subprocess.call([python(), *args, *extra], cwd=ROOT,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})


if __name__ == "__main__":
    sys.exit(main())
