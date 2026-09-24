"""螢幕辨識與觀測狀態轉換。"""
import os

# 模板比對的矩陣很小，OpenBLAS 多執行緒在 Windows 上喚醒成本遠大於計算本身，
# 實測單執行緒快約 250 倍（一幀 4 秒 → 0.02 秒）。必須在 numpy 載入前設定。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
