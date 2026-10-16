import pytest

from emote.comfy import ComfyClient, fill_workflow
from emote.sprite import EmoteError


def test_fill_keeps_types_for_whole_placeholders_and_stringifies_inline():
    wf = {"1": {"inputs": {"seed": "{{seed}}", "text": "a {{pose}} pigeon, seed {{seed}}", "w": [3, "{{n}}"]}}}
    out = fill_workflow(wf, {"seed": 42, "pose": "flying", "n": 7})
    assert out["1"]["inputs"] == {"seed": 42, "text": "a flying pigeon, seed 42", "w": [3, 7]}
    assert wf["1"]["inputs"]["seed"] == "{{seed}}"  # input untouched


def test_missing_placeholder_is_a_clear_error():
    with pytest.raises(EmoteError, match="no value"):
        fill_workflow({"a": "{{nope}}"}, {"seed": 1})


def test_unreachable_server_error_is_clear():
    c = ComfyClient(port=1)  # nothing listens here
    assert not c.alive()
    with pytest.raises(EmoteError, match="Cannot reach ComfyUI"):
        c.system_stats()
