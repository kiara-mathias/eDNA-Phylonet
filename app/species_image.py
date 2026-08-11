"""Reference photos for a resolved species/genus name.

Looks up a freely attributed still image from iNaturalist (taxon default
photo), then GBIF (species media, then an occurrence still). Failures and
missing media become a placeholder silhouette — rarer names often have
nothing usable. ``get_species_image`` is ``st.cache_data`` so Streamlit
reruns do not re-hit the APIs.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from typing import TYPE_CHECKING

import requests
import streamlit as st

from theme.tokens import NAVY, SEAFOAM

if TYPE_CHECKING:
    from src.fallback.novelty import FallbackPrediction

logger = logging.getLogger(__name__)

INAT_TAXA_URL = "https://api.inaturalist.org/v1/taxa"
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
GBIF_MEDIA_URL = "https://api.gbif.org/v1/species/{key}/media"
GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
TIMEOUT_SECONDS = 8.0
USER_AGENT = "edna-classifier/0.1 (eDNA specimen record; reference photos)"
_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

HttpGet = Callable[..., Any]


class _Response(Protocol):
    def raise_for_status(self) -> None: ...
    def json(self) -> Any: ...


@dataclass(frozen=True)
class SpeciesImage:
    url: str
    attribution: str
    source: str
    scientific_name: str
    license: str | None = None


PLACEHOLDER_SVG = f"""
<svg class="silhouette" viewBox="0 0 240 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="No reference photo">
  <rect width="240" height="150" rx="8" fill="{NAVY}"/>
  <path fill="{SEAFOAM}" fill-opacity="0.82"
        d="M38 78c8-28 62-46 112-18 12 7 22 8 32 4l28-18v50l-28-16c-10-4-20-3-32 4-50 28-104 10-112-18z"/>
  <circle cx="78" cy="70" r="4.5" fill="{NAVY}"/>
</svg>
"""


def reference_taxon_name(prediction: FallbackPrediction) -> str:
    """Species if the cascade committed there; otherwise nearest known species/genus."""
    if prediction.species:
        return str(prediction.species)
    if prediction.closest_relative_species:
        return str(prediction.closest_relative_species)
    if prediction.genus:
        return str(prediction.genus)
    return ""


def _default_http_get(url: str, **kwargs) -> _Response:
    kwargs.setdefault("timeout", TIMEOUT_SECONDS)
    headers = dict(_HEADERS)
    headers.update(kwargs.pop("headers", {}))
    return requests.get(url, headers=headers, **kwargs)


def _best_inat_taxon(results: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    needle = name.strip().lower()
    if not results:
        return None
    exact = [row for row in results if str(row.get("name") or "").lower() == needle]
    pool = exact or results
    with_photo = [row for row in pool if (row.get("default_photo") or {}).get("medium_url")]
    return (with_photo or pool)[0]


def _from_inaturalist(name: str, http_get: HttpGet) -> SpeciesImage | None:
    response = http_get(
        INAT_TAXA_URL,
        params={"q": name, "is_active": "true", "per_page": 8},
    )
    response.raise_for_status()
    taxon = _best_inat_taxon(list(response.json().get("results") or []), name)
    if not taxon:
        return None
    photo = taxon.get("default_photo") or {}
    url = photo.get("medium_url") or photo.get("square_url")
    if not url:
        return None
    attribution = str(photo.get("attribution") or "iNaturalist").strip()
    return SpeciesImage(
        url=str(url),
        attribution=attribution,
        source="iNaturalist",
        scientific_name=str(taxon.get("name") or name),
        license=photo.get("license_code"),
    )


def _from_gbif(name: str, http_get: HttpGet) -> SpeciesImage | None:
    match = http_get(GBIF_MATCH_URL, params={"name": name})
    match.raise_for_status()
    body = match.json()
    key = body.get("usageKey")
    if not key or body.get("matchType") == "NONE":
        return None
    canonical = str(body.get("canonicalName") or name)

    media = http_get(GBIF_MEDIA_URL.format(key=key), params={"limit": 5})
    media.raise_for_status()
    for item in media.json().get("results") or []:
        url = item.get("identifier")
        if not url:
            continue
        credit = item.get("rightsHolder") or item.get("creator") or "GBIF"
        return SpeciesImage(
            url=str(url),
            attribution=str(credit),
            source="GBIF",
            scientific_name=canonical,
            license=item.get("license"),
        )

    occ = http_get(
        GBIF_OCCURRENCE_URL,
        params={"taxonKey": key, "mediaType": "StillImage", "limit": 1},
    )
    occ.raise_for_status()
    records = occ.json().get("results") or []
    if not records:
        return None
    for item in records[0].get("media") or []:
        url = item.get("identifier")
        if not url:
            continue
        credit = item.get("rightsHolder") or item.get("creator") or "GBIF"
        return SpeciesImage(
            url=str(url),
            attribution=str(credit),
            source="GBIF",
            scientific_name=canonical,
            license=item.get("license"),
        )
    return None


def lookup_species_image(scientific_name: str, http_get: HttpGet | None = None) -> SpeciesImage | None:
    """Query iNaturalist, then GBIF. Returns None when nothing usable is found."""
    name = scientific_name.strip()
    if not name:
        return None
    getter = http_get or _default_http_get
    try:
        found = _from_inaturalist(name, getter)
        if found:
            return found
        return _from_gbif(name, getter)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        logger.warning("Species image lookup failed for %s: %s", name, exc)
        return None


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_species_image(scientific_name: str) -> SpeciesImage | None:
    return lookup_species_image(scientific_name)


def render_reference_photo(prediction: FallbackPrediction) -> None:
    name = reference_taxon_name(prediction)
    image = get_species_image(name) if name else None
    role = "nearest known" if prediction.is_novel and not prediction.species else "reference"

    st.markdown('<p class="panel-kicker">Reference</p>', unsafe_allow_html=True)
    if image is None:
        st.markdown(
            f'<div class="ref-photo">{PLACEHOLDER_SVG}'
            f'<p class="mono-quiet">No reference photo</p></div>',
            unsafe_allow_html=True,
        )
        return

    st.image(image.url, width="stretch")
    caption = f"{role} · {image.scientific_name} · {image.source} · {image.attribution}"
    st.markdown(f'<p class="mono-quiet">{html.escape(caption)}</p>', unsafe_allow_html=True)
