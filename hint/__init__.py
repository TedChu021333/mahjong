"""提示模式：辨識畫面、由 AI 給建議，人自己點。"""
import os

# 見 perception/__init__.py：OpenBLAS 單執行緒，必須在 numpy 載入前設定
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
