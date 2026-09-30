from game.rules import Action, ActionType
from game.tiles import parse
from control.actuator import Actuator
from hint.__main__ import MAX_RETRIES, RETRY_AFTER, run
from hint.advisor import Observation
from perception.config import ACTION_BUTTON_CENTERS, CANCEL_BUTTON_REGION
from perception.continue_button import ContinueButton
from perception.hand_reader import _cell_box, drawn_tile_box
from perception.table_reader import ClaimTile


def P(value: str) -> int:
    return parse(value)[0]


def center(box):
    left, top, right, bottom = box
    return ((left + right) // 2, (top + bottom) // 2)


def my_turn(buttons=None) -> Observation:
    hand = tuple(parse("123456789m123p45p11z9s"))
    boxes = tuple(_cell_box(index) for index in reversed(range(16))) + (drawn_tile_box(),)
    return Observation(hand, buttons, None, (), boxes)


def recorder():
    clicks = []
    return clicks, Actuator(click=clicks.append, sleep=lambda _: None)


def test_discard_clicks_the_drawn_tile_first():
    clicks, actuator = recorder()
    assert actuator.perform(Action(ActionType.DISCARD, tile=P("9s")), my_turn())
    assert clicks == [center(drawn_tile_box())]


def test_discard_clicks_rightmost_copy_in_grid():
    clicks, actuator = recorder()
    actuator.perform(Action(ActionType.DISCARD, tile=P("1z")), my_turn())
    assert clicks == [center(_cell_box(0))]


def test_declare_presses_button_then_tile():
    clicks, actuator = recorder()
    actuator.perform(Action(ActionType.DECLARE, tile=P("9s")), my_turn(frozenset({"declare"})))
    assert clicks == [ACTION_BUTTON_CENTERS["declare"], center(drawn_tile_box())]


def test_claims_and_pass_use_buttons_only_when_shown():
    clicks, actuator = recorder()
    waiting = Observation(tuple(parse("11m123456789p139s57z")), frozenset({"pung"}),
                          ClaimTile(P("1m"), 0.9, "right"), ())
    assert actuator.perform(Action(ActionType.PUNG, tile=P("1m")), waiting)
    assert not actuator.perform(Action(ActionType.WIN, tile=P("1m")), waiting)  # 胡沒亮
    assert actuator.perform(Action(ActionType.PASS), waiting)
    left, top, right, bottom = CANCEL_BUTTON_REGION
    assert clicks == [ACTION_BUTTON_CENTERS["pung"], ((left + right) // 2, (top + bottom) // 2)]


def test_missing_tile_is_not_clicked():
    clicks, actuator = recorder()
    assert not actuator.perform(Action(ActionType.DISCARD, tile=P("5z")), my_turn())
    assert clicks == []


def test_auto_loop_acts_once_per_screen_and_retries_later():
    clicks, actuator = recorder()
    observation = my_turn()
    now = [0.0]

    def frames():
        for step in range(40):
            now[0] = step * 0.3
            yield f"{step}", None

    run(frames(), readers=None, actuator=actuator,
        observe_fn=lambda frame, readers: observation, clock=lambda: now[0],
        save_result=lambda frame, prefix="": "x", log=lambda line, detail=None, echo=True: None)
    # 第一次點擊後，每隔 RETRY_AFTER 秒補點一次，最多 MAX_RETRIES 次
    assert len(clicks) == 1 + MAX_RETRIES
    assert RETRY_AFTER > 0.3


def swap_screen(raised=(), confirm=False) -> Observation:
    hand = tuple(parse("244m388p35588s12345z"))
    boxes = tuple(_cell_box(index) for index in reversed(range(16)))
    flags = tuple(index in raised for index in range(16))
    return Observation(hand, None, None, (), boxes, confirm, flags)


def test_swap_selects_tiles_one_click_at_a_time_then_confirms():
    from game.rules import ActionType
    action = Action(ActionType.SWAP, tiles=tuple(parse("123z")))  # 東南西在第 11-13 張
    clicks, actuator = recorder()
    actuator.perform(action, swap_screen())
    assert clicks == [center(_cell_box(4))]          # 先選東（第 11 張）
    clicks.clear()
    actuator.perform(action, swap_screen(raised={11, 12, 13, 0}))
    assert clicks == [center(_cell_box(15))]         # 取消多選的第 0 張
    clicks.clear()
    actuator.perform(action, swap_screen(raised={11, 12, 13}, confirm=True))
    from perception.config import SWAP_CONFIRM_BUTTON
    assert clicks == [SWAP_CONFIRM_BUTTON]


def test_swap_nothing_presses_keep():
    from perception.config import SWAP_KEEP_BUTTON
    clicks, actuator = recorder()
    actuator.perform(Action(ActionType.SWAP), swap_screen())
    assert clicks == [SWAP_KEEP_BUTTON]


def test_multiple_chow_options_click_chow_then_the_matching_panel():
    # 上家打 6s，手上 4s5s7s：可吃 456s 或 567s（選項框依此順序）
    hand = tuple(parse("111m222m333m99p457s11z"))
    assert len(hand) == 16
    waiting = Observation(hand, frozenset({"chow"}), ClaimTile(P("6s"), 0.9, "left"), ())
    clicks, actuator = recorder()
    action = Action(ActionType.CHOW, tile=P("6s"), tiles=tuple(parse("567s")))
    assert actuator.perform(action, waiting)
    assert clicks == [ACTION_BUTTON_CENTERS["chow"]]
    panels = ((850, 720), (1212, 720))
    with_panels = Observation(hand, None, None, (), chow_panels=panels)
    assert actuator.follow_up(with_panels)
    assert clicks[-1] == (1212, 720)
    assert not actuator.follow_up(with_panels)  # 只點一次


def test_single_chow_option_needs_no_panel():
    hand = tuple(parse("111m222m333m99p45s111z"))
    assert len(hand) == 16
    waiting = Observation(hand, frozenset({"chow"}), ClaimTile(P("6s"), 0.9, "left"), ())
    clicks, actuator = recorder()
    actuator.perform(Action(ActionType.CHOW, tile=P("6s"), tiles=tuple(parse("456s"))), waiting)
    assert actuator.pending_chow is None


def test_auto_mode_requires_confirming_a_computer_only_room():
    from hint.__main__ import confirm_training_room
    assert confirm_training_room(ask=lambda _: "y")
    assert confirm_training_room(ask=lambda _: " Y ")
    assert not confirm_training_room(ask=lambda _: "")
    assert not confirm_training_room(ask=lambda _: "n")


def test_auto_loop_presses_continue_then_next_game_after_a_delay():
    from perception.config import NEXT_GAME_CLICKS, NEXT_GAME_DELAY, NEXT_GAME_REGION

    clicks, actuator = recorder()
    button = ContinueButton("小結算繼續", (1459, 977))
    next_game = center(NEXT_GAME_REGION)
    now = [0.0]
    # 每幀 0.5 秒：結算畫面 → 讀不到手牌 11 秒 → 新的一局
    screens = [button] * 2 + [None] * 22 + [my_turn()] * 2 + [None] * 10

    def frames():
        for step, screen in enumerate(screens):
            now[0] = step * 0.5
            yield f"{step}", screen

    saved = []
    run(frames(), readers=None, actuator=actuator, observe_fn=lambda frame, readers: frame
        if isinstance(frame, Observation) else None,
        find_continue=lambda frame: frame if isinstance(frame, ContinueButton) else None,
        clock=lambda: now[0], save_result=lambda frame: saved.append(frame) or "result/x.png")
    assert saved == [button]                          # 每局只存一張結算截圖
    assert clicks[0] == (1459, 977)                   # 按繼續
    assert clicks[1] == next_game                     # 3 秒後按下一場
    assert clicks[1:1 + NEXT_GAME_CLICKS] == [next_game] * NEXT_GAME_CLICKS
    assert clicks[1 + NEXT_GAME_CLICKS:] == [center(drawn_tile_box())]  # 新局開始後不再點
    assert NEXT_GAME_DELAY == 3.0


def test_save_result_writes_a_png(tmp_path, monkeypatch):
    import numpy as np
    import hint.__main__ as hint_main
    from perception.capture import read_image

    monkeypatch.setattr(hint_main, "RESULT_DIR", tmp_path / "result")
    path = hint_main.save_result(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert path.parent == tmp_path / "result" and path.suffix == ".png"
    assert read_image(path).shape == (1080, 1920, 3)
