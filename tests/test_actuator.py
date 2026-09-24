from game.rules import Action, ActionType
from game.tiles import parse
from control.actuator import Actuator
from hint.__main__ import MAX_RETRIES, RETRY_AFTER, run
from hint.advisor import Observation
from perception.config import ACTION_BUTTON_CENTERS, CANCEL_BUTTON_REGION
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
        observe_fn=lambda frame, readers: observation, clock=lambda: now[0])
    # 第一次點擊後，每隔 RETRY_AFTER 秒補點一次，最多 MAX_RETRIES 次
    assert len(clicks) == 1 + MAX_RETRIES
    assert RETRY_AFTER > 0.3
