"""Offline unit tests for src.ingest.bold_client.BoldClient.

All ``requests`` calls are mocked, so these run without any network access
and without a live BOLD API dependency. The NDJSON streaming behavior
tested here (no server-side pagination; one JSON object per line) was
confirmed against the real BOLD Portal API before being encoded here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.ingest.bold_client import BoldAPIError, BoldClient


def _mock_response(json_data) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _mock_stream_response(records: list[dict]) -> MagicMock:
    """Build a mock response mimicking BOLD's NDJSON download stream."""
    import json

    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.iter_lines.return_value = iter(json.dumps(r) for r in records)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


@pytest.fixture
def session():
    return MagicMock()


@pytest.fixture
def client(session):
    return BoldClient(base_url="https://example.test/api", session=session, timeout_seconds=5)


def test_preprocess_query_calls_correct_endpoint(client, session):
    session.get.return_value = _mock_response({"successful_terms": [{"submitted": "tax:class:Teleostei"}]})

    result = client.preprocess_query("tax:class:Teleostei")

    session.get.assert_called_once_with(
        "https://example.test/api/query/preprocessor",
        params={"query": "tax:class:Teleostei"},
        timeout=5,
    )
    assert result["successful_terms"][0]["submitted"] == "tax:class:Teleostei"


def test_submit_query_extracts_query_id(client, session):
    session.get.return_value = _mock_response({"query_id": "abc123"})

    query_id, raw = client.submit_query("tax:family:Gadidae", extent="full")

    session.get.assert_called_once_with(
        "https://example.test/api/query",
        params={"query": "tax:family:Gadidae", "extent": "full"},
        timeout=5,
    )
    assert query_id == "abc123"
    assert raw == {"query_id": "abc123"}


@pytest.mark.parametrize("key", ["query_id", "queryId", "id", "token"])
def test_submit_query_accepts_alternate_id_field_names(client, session, key):
    session.get.return_value = _mock_response({key: "xyz"})

    query_id, _ = client.submit_query("tax:family:Gadidae")

    assert query_id == "xyz"


def test_submit_query_raises_on_missing_id(client, session):
    session.get.return_value = _mock_response({"unexpected": "shape"})

    with pytest.raises(BoldAPIError):
        client.submit_query("tax:family:Gadidae")


def test_stream_documents_parses_ndjson_lines(client, session):
    records = [{"processid": "P1"}, {"processid": "P2"}, {"processid": "P3"}]
    session.get.return_value = _mock_stream_response(records)

    result = list(client.stream_documents("qid"))

    assert result == records
    session.get.assert_called_once_with(
        "https://example.test/api/documents/qid/download",
        params={"format": "json"},
        stream=True,
        timeout=5,
    )


def test_stream_documents_skips_blank_lines(client, session):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.iter_lines.return_value = iter(['{"processid": "P1"}', "", "  ", '{"processid": "P2"}'])
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    session.get.return_value = resp

    result = list(client.stream_documents("qid"))

    assert result == [{"processid": "P1"}, {"processid": "P2"}]


def test_stream_documents_raises_on_malformed_line(client, session):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.iter_lines.return_value = iter(["not-json"])
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    session.get.return_value = resp

    with pytest.raises(BoldAPIError):
        list(client.stream_documents("qid"))


def test_fetch_all_runs_full_three_stage_flow(client, session):
    session.get.side_effect = [
        _mock_response({"successful_terms": []}),  # preprocessor
        _mock_response({"query_id": "qid42"}),  # submit
        _mock_stream_response([{"processid": "P1"}]),  # download stream
    ]

    query_id, record_iter = client.fetch_all("tax:family:Gadidae")
    records = list(record_iter)

    assert query_id == "qid42"
    assert records == [{"processid": "P1"}]
    assert session.get.call_count == 3
