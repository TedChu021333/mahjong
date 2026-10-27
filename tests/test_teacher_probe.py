from collections import Counter
from random import Random

from game.tiles import is_flower
from training.teacher_probe import Point, StreamWall, determinize, evaluate, report, sample_points, summarize


def hidden_tiles(state, player):
    tiles = Counter(state.wall)
    for opponent in range(4):
        if opponent != player:
            tiles.update({tile: n for tile, n in enumerate(state.players[opponent].hand) if n})
    return tiles


def test_determinize_redeals_only_what_the_player_cannot_see():
    point = sample_points(1, seed=5)[0]
    world = determinize(point.state, point.player, Random(1))
    assert world.players[point.player].hand == point.state.players[point.player].hand
    assert hidden_tiles(world, point.player) == hidden_tiles(point.state, point.player)
    for opponent in range(4):
        hand = world.players[opponent].hand
        assert sum(hand) == sum(point.state.players[opponent].hand)
        assert not any(n and is_flower(tile) for tile, n in enumerate(hand))
    assert world.drawable == point.state.drawable


def test_stream_wall_gives_each_player_their_own_pile():
    class State:
        current_player = 1

    state = State()
    wall = StreamWall(state, list(range(8)))
    assert len(wall) == 8 and sorted(wall) == list(range(8))
    assert wall.pop() == 5          # 第 1 家那疊是 1、5
    assert wall.pop() == 1
    assert wall.pop() == 4          # 自己那疊摸完，拿別疊最多的（一樣多取第一疊）
    assert len(wall) == 5


def test_same_candidate_twice_gives_identical_results():
    point = sample_points(1, seed=7)[0]
    same = Point(point.state, point.player, [point.candidates[0]] * 2)
    for streams in (False, True):
        rows = evaluate(same, rollouts=3, seed=2, streams=streams)
        assert len(rows) == 3 and all(row[0] == row[1] for row in rows)
    text = report([summarize(same, rows)])
    assert "100% 的候選對每次結果完全一樣" in text


def test_other_policy_keeps_only_points_where_it_discards_something_else():
    from game.rules import Action, ActionType, legal_actions

    def other(state, player, rng):
        discards = sorted({a.tile for a in legal_actions(state, player) if a.kind == ActionType.DISCARD})
        return Action(ActionType.DISCARD, discards[-1])

    points = sample_points(3, seed=4, other=other)
    assert len(points) == 3
    for point in points:
        assert len(point.candidates) == 2 and point.candidates[0] != point.candidates[1]
        assert point.candidates[1] == other(point.state, point.player, None).tile
