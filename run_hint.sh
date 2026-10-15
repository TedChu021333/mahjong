#!/usr/bin/env bash
# 學習型 AI 提示模式：只給建議、不自動點擊（真人配對的場用這個）。
# 用法：./run_hint.sh [其他參數]
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8
exec .venv-train/Scripts/python -m hint --ai nn --model models/rl3_0960.pt "$@"
