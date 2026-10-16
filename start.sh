#!/usr/bin/env bash
# 所有執行指令的總選單（麻將、大老二）。用法：./start.sh 或 ./start.sh <編號> [其他參數]
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8
exec .venv-train/Scripts/python start.py "$@"
