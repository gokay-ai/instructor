"""xAI handlers raise ResponseParsingError for unparseable completions (#3083)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.errors import ResponseParsingError
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.retry import retry_sync_v2


class User(BaseModel):
    name: str


def _empty_response() -> SimpleNamespace:
    return SimpleNamespace(tool_calls=[], text="", content="")


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        (Mode.TOOLS, "No tool calls returned from xAI"),
        (Mode.JSON_SCHEMA, "Could not parse xAI response"),
        (Mode.MD_JSON, "Could not extract JSON from xAI response"),
    ],
)
def test_empty_xai_response_raises_response_parsing_error(
    mode: Mode, message: str
) -> None:
    handlers = mode_registry.get_handlers(Provider.XAI, mode)
    response = _empty_response()

    with pytest.raises(ResponseParsingError, match=message) as exc_info:
        handlers.response_parser(response=response, response_model=User, stream=False)

    assert exc_info.value.mode == mode.name
    assert exc_info.value.raw_response is response
    # Existing ``except ValueError`` callers keep working.
    assert isinstance(exc_info.value, ValueError)


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON])
def test_empty_xai_response_is_reasked_under_max_retries(mode: Mode) -> None:
    responses = [
        _empty_response(),
        SimpleNamespace(tool_calls=[], text='{"name": "Ada"}', content=""),
    ]
    calls = 0

    def fake_create(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return responses.pop(0)

    result = retry_sync_v2(
        func=fake_create,
        response_model=User,
        provider=Provider.XAI,
        mode=mode,
        context=None,
        max_retries=2,
        args=(),
        kwargs={"messages": [{"role": "user", "content": "Extract the user"}]},
        strict=None,
        hooks=None,
    )

    assert result.name == "Ada"
    assert calls == 2
