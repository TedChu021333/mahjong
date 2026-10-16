import importlib.util
from pathlib import Path

import start

ROOT = Path(__file__).resolve().parent.parent


def load_launcher():
    spec = importlib.util.spec_from_file_location("launcher", ROOT / "launcher.pyw")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_button_runs_start_menu_entry_in_new_window():
    launcher = load_launcher()
    cmd = launcher.command(4)
    assert cmd[:2] == ["cmd", "/k"]                 # 程式結束後視窗留著，看得到記錄與錯誤
    assert cmd[-2:] == [str(start.ROOT / "start.py"), "4"]
