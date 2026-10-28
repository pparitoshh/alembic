"""Issue #114 follow-up: new traces can use the opt-in mock v2; every trace is replayed with the mock it was made with."""
from contextlib import contextmanager

import pytest

from distillkit import toolcheck
from distillkit.config import GenerateCfg
from distillkit.tools import MOCK_VERSION, MOCK_VERSION_V2


@pytest.mark.parametrize("row_version, replayed", [(None, MOCK_VERSION), (MOCK_VERSION, MOCK_VERSION),
                                                   (MOCK_VERSION_V2, MOCK_VERSION_V2)])
def test_check_trace_replays_with_the_rows_mock(monkeypatch, row_version, replayed):
    used = []

    @contextmanager
    def session(valid_flags=None, *, job_ids=None, mock_version=MOCK_VERSION):
        used.append(mock_version)
        yield

    monkeypatch.setattr(toolcheck, "mock_session", session)
    monkeypatch.setattr(toolcheck, "_check_trace", lambda row: {"passed": True})
    row = {} if row_version is None else {"mock_version": row_version}
    assert toolcheck.check_trace(row) == {"passed": True} and used == [replayed]


def test_unknown_mock_version_is_still_rejected():
    res = toolcheck.check_trace({"mock_version": "made-up-v9", "messages": []})
    assert not res["passed"] and any("made-up-v9" in e for e in res["trace_errors"])


def test_generate_config_defaults_to_v1_and_accepts_v2():
    assert GenerateCfg.model_fields["mock_version"].default == MOCK_VERSION
    field = GenerateCfg.model_fields["mock_version"].annotation
    assert MOCK_VERSION_V2 in field.__args__
