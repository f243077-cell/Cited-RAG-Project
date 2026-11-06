import asyncio

import pytest

from citerag.errors import LLMUnavailableError
from citerag.llm.fake import FakeGateway
from citerag.schemas import LLMCitedAnswer, StandaloneQuery


def _call(gateway, call_name="condense", schema=StandaloneQuery, **kw):
    args = {"role": "fast", "schema": schema, "system": "s", "user": "u", "call_name": call_name}
    return asyncio.run(gateway.structured_call(**{**args, **kw}))


def test_returns_queued_objects_in_order_and_records_calls():
    gw = FakeGateway()
    gw.queue("condense", StandaloneQuery(query="first"))
    gw.queue("condense", StandaloneQuery(query="second"))
    assert _call(gw, request_id="r1").parsed.query == "first"
    out = _call(gw)
    assert out.parsed.query == "second" and out.requests == 1 and out.model == "fake/fast"
    assert [c.call_name for c in gw.calls] == ["condense", "condense"]
    assert gw.calls[0].request_id == "r1" and gw.calls[0].schema is StandaloneQuery


def test_raises_on_demand_and_still_records():
    gw = FakeGateway()
    gw.queue("generate", LLMUnavailableError("down"))
    with pytest.raises(LLMUnavailableError):
        _call(gw, "generate", LLMCitedAnswer, role="strong")
    assert len(gw.calls) == 1


def test_unqueued_call_and_wrong_schema_fail_loudly():
    gw = FakeGateway()
    with pytest.raises(LookupError):
        _call(gw)
    gw.queue("condense", StandaloneQuery(query="x"))
    with pytest.raises(TypeError):
        _call(gw, schema=LLMCitedAnswer)
