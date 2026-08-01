"""Thin client for the BOLD Systems Portal API.

Implements the documented three-stage query flow
(https://boldsystems.org/data/api/?type=webservices):

1. ``GET /api/query/preprocessor`` -- validates a search query string
   (``scope:term`` or ``scope:field:term`` phrases, e.g.
   ``tax:class:Teleostei``; combine same-scope phrases with ``;`` for OR,
   e.g. ``tax:family:Gadidae;tax:family:Scombridae``) against BOLD's
   controlled vocabularies and returns formal query triplets.
2. ``GET /api/query`` -- submits the query (with an ``extent``, e.g.
   ``full``) and returns a ``query_id`` token valid for 24 hours.
3. ``GET /api/documents/<query_id>/download`` -- streams the matching
   BCDM-schema records as newline-delimited JSON (one JSON object per
   line), confirmed by live testing against the real API. There is no
   client-controlled pagination on this endpoint (``start``/``size`` query
   params are accepted but ignored) -- the "batches of 1,000" language in
   BOLD's docs refers to server-side chunked transfer of the single
   response stream, not request-level paging. Consumers should read the
   stream (``stream_documents``) and chunk it into files client-side if
   desired, as ``fetch_bold.py`` does.

The BOLD Portal API has no scope for filtering by marker code at query time
(scopes are ``tax``, ``bin``, ``geo``, ``inst``, ``recordsetcode``, ``ids``),
so marker filtering (e.g. keeping only COI-5P) is done client-side after
download, in ``fetch_bold.py`` / ``clean.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://portal.boldsystems.org/api"
DEFAULT_TIMEOUT_SECONDS = 60


class BoldAPIError(RuntimeError):
    """Raised when the BOLD API returns an unexpected or invalid response."""


class BoldClient:
    """Minimal client for the three-stage BOLD Portal API query flow."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def preprocess_query(self, query: str) -> dict[str, Any]:
        """Validate ``query`` and resolve it into formal query triplets."""
        resp = self.session.get(
            f"{self.base_url}/query/preprocessor",
            params={"query": query},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()

    def submit_query(self, query: str, extent: str = "full") -> tuple[str, dict[str, Any]]:
        """Submit ``query`` and return ``(query_id, raw_response)``."""
        resp = self.session.get(
            f"{self.base_url}/query",
            params={"query": query, "extent": extent},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        query_id = self._extract_query_id(data)
        if not query_id:
            raise BoldAPIError(
                f"BOLD /api/query response did not contain a recognizable "
                f"query id field: {data!r}"
            )
        return query_id, data

    @staticmethod
    def _extract_query_id(data: dict[str, Any]) -> str | None:
        for key in ("query_id", "queryId", "id", "token"):
            value = data.get(key)
            if value:
                return value
        return None

    def stream_documents(self, query_id: str, fmt: str = "json") -> Iterator[dict[str, Any]]:
        """Stream BCDM records for ``query_id`` one at a time.

        The download endpoint returns the full matching result set as
        newline-delimited JSON in a single (chunked-transfer) response;
        this streams and parses it line by line rather than buffering the
        whole response in memory.
        """
        with self.session.get(
            f"{self.base_url}/documents/{query_id}/download",
            params={"format": fmt},
            stream=True,
            timeout=self.timeout_seconds,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BoldAPIError(f"Failed to parse NDJSON line from BOLD: {exc}") from exc

    def fetch_all(
        self,
        query: str,
        extent: str = "full",
        fmt: str = "json",
    ) -> tuple[str, Iterator[dict[str, Any]]]:
        """Run the full preprocess -> submit -> download flow for ``query``.

        Returns ``(query_id, record_iterator)``.
        """
        self.preprocess_query(query)
        query_id, _ = self.submit_query(query, extent=extent)
        return query_id, self.stream_documents(query_id, fmt=fmt)
