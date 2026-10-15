#!/usr/bin/env bash
# 學習型 AI 自動模式：只在對手全是電腦的「訓練場」使用（啟動時要輸入 y 確認）。
# 用法：./run.sh [其他參數，例如 --efficiency-turns 6 或 --model models/xxx.pt]
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8
exec .venv-train/Scripts/python -m hint --auto --ai nn --model models/rl3_0960.pt "$@"
