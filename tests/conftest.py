import os

# 見 perception/__init__.py：測試模組會先 import numpy，所以在這裡先設定
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
