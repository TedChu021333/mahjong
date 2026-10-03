from control.window import find_game_window, focus_game, game_is_foreground, user_idle_seconds


def test_missing_window_is_handled():
    title = "不存在的視窗標題_7f3a"
    assert find_game_window(title) is None
    assert game_is_foreground(title) is None
    assert focus_game(title) is False


def test_idle_seconds_is_not_negative():
    assert user_idle_seconds() >= 0
