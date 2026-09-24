import pytest

from game.tiles import parse, tile_code


def test_tile_code_is_inverse_of_parse():
    for tile in range(42):
        assert parse(tile_code(tile)) == [tile]
    assert tile_code(4) == "5m"
    assert tile_code(27) == "1z"


def test_tile_code_rejects_invalid_tile():
    with pytest.raises(ValueError):
        tile_code(42)
