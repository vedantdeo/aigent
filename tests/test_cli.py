from __future__ import annotations

import pytest

from aigent.cli import MODES, find_mode, main, menu_text


def test_five_modes_in_menu_order() -> None:
    assert [m.key for m in MODES] == ["call", "stream", "extract", "loop", "chat"]


@pytest.mark.parametrize(
    ("name", "key"), [("chat", "chat"), ("5", "chat"), (" Loop ", "loop"), ("1", "call")]
)
def test_find_mode_by_key_or_number(name: str, key: str) -> None:
    mode = find_mode(name)
    assert mode is not None and mode.key == key


@pytest.mark.parametrize("name", ["nope", "0", "6", ""])
def test_find_mode_rejects_unknown(name: str) -> None:
    assert find_mode(name) is None


def test_menu_lists_every_mode() -> None:
    text = menu_text()
    for number, mode in enumerate(MODES, start=1):
        assert f"{number}. {mode.key}" in text


def test_unknown_mode_argument_exits_with_menu() -> None:
    with pytest.raises(SystemExit, match="unknown mode 'nope'"):
        main(["nope"])
