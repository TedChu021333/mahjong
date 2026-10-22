import start


def test_choose_by_argument_passes_extra_args():
    index, extra = start.choose(["1", "--efficiency-turns", "6"])
    assert start.MENU[index][0].startswith("麻將：自動模式") and extra == ["--efficiency-turns", "6"]


def test_choose_from_menu_and_exit():
    index, _ = start.choose([], ask=lambda prompt: "5")
    assert start.MENU[index][1] == ["-m", "bigtwo.live"]
    assert start.choose([], ask=lambda prompt: "0") == (None, [])
    assert start.choose([], ask=lambda prompt: "abc") == (None, [])


def test_every_entry_runs_a_module():
    for name, args in start.MENU:
        assert args[0] == "-m"
