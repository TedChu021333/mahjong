from hint.hand_end import DRAW_TEXT, MISSED_TEXT, HandBook


def test_dealer_is_logged_once_per_hand():
    book = HandBook()
    assert book.see_dealer(None) == []
    assert book.see_dealer("top") == ["本局莊家：對家"]
    assert book.see_dealer("top") == []


def test_unsettled_hand_with_the_same_dealer_is_a_draw():
    book = HandBook()
    book.see_dealer("top")
    book.hand_ended(settled=False)
    messages = book.see_dealer("top")
    assert any(DRAW_TEXT in m for m in messages)


def test_unsettled_hand_with_a_new_dealer_was_a_missed_settlement():
    # 10/19 15:19：門風從西變南（對家 → 上家當莊）、金幣沒變
    book = HandBook()
    book.see_dealer("top")
    book.hand_ended(settled=False)
    messages = book.see_dealer("left")
    assert any(MISSED_TEXT in m for m in messages)


def test_settled_hand_needs_no_judgement():
    book = HandBook()
    book.see_dealer("top")
    book.hand_ended(settled=True)
    assert book.see_dealer("left") == ["本局莊家：上家"]


def test_unknown_previous_dealer_cannot_be_judged():
    book = HandBook()
    book.hand_ended(settled=False)
    messages = book.see_dealer("me")
    assert len(messages) == 2 and DRAW_TEXT not in messages[1] and MISSED_TEXT not in messages[1]
