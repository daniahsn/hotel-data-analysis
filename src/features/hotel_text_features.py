"""Text-derived features from hotel listing fields (attractions, facilities tokens/keywords)."""

from __future__ import annotations

import re

import pandas as pd

# "within 3000 metre" / "Place : within 3000 metre"
_METRE_ATTR = re.compile(r"within\s+\d+\s*metre", re.IGNORECASE)
# Hyphen, en dash, em dash before distances (vendor HTML varies)
_KM_MI_DISTANCE = re.compile(
    r"[-\u2013\u2014]\s*\d+\.?\d*\s*(?:km|mi)\b",
    re.IGNORECASE,
)
_BR_SPLIT = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _strip_html_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s)


def _count_attractions_one(raw: object) -> int:
    """
    Heuristic count of nearby POIs from free-text ``Attractions``.

    Supports common vendor shapes:

    - HTML lists with ``<br />`` and ``Name - 1.8 km / 1.1 mi`` lines
    - Prose like ``Place : within 3000 metre  Other : within 4000 metre``
    - Simple comma-separated lists (fallback)

    If HTML is present but line-based counting finds nothing (odd encoding), falls back to
    counting ``- X km`` / ``- X mi`` spans in the string and drops the trailing ``preferred airport`` block.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return 0
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return 0

    low = s.lower()
    # "Berat : within 130000 metre  Tower ..." (plain text, no <br>)
    if "<br" not in low and _METRE_ATTR.search(s):
        return len(_METRE_ATTR.findall(s))

    # HTML distance blocks (Kaggle / booking-style)
    if "<br" in low or "<p>" in low or "</p>" in low:
        n = 0
        for part in _BR_SPLIT.split(s):
            t = _strip_html_tags(part)
            t = re.sub(r"\s+", " ", t).strip()
            if len(t) < 10:
                continue
            tl = t.lower()
            if "preferred airport" in tl or ("airport for" in tl and "intl" in tl):
                continue
            if _KM_MI_DISTANCE.search(t) or _METRE_ATTR.search(t):
                n += 1
        if n > 0:
            return n
        body = s
        if "preferred airport" in low:
            body = s[: low.rfind("preferred airport")]
        fallback = len(_KM_MI_DISTANCE.findall(body))
        if fallback > 0:
            return fallback

    comma_parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(comma_parts) > 1 or (len(comma_parts) == 1 and "," in s):
        return len(comma_parts)
    return 0


def attractions_count(series: pd.Series) -> pd.Series:
    """Derive ``attractions_count`` from messy ``Attractions`` text (HTML, metre lists, or commas)."""
    s = series.fillna("").astype(str).str.strip()
    empty = s.eq("") | s.str.lower().eq("nan")
    counts = s.map(_count_attractions_one)
    return counts.where(~empty, 0).astype("Int64")


# --- HotelFacilities: token count + amenity keyword hits ----------------------------

# Curated common amenity phrases (lowercase). Longer phrases first so sub-phrases still match separately.
_RAW_FACILITY_KEYWORDS: tuple[str, ...] = (
    "24-hour front desk",
    "airport transportation",
    "airport pick up",
    "airport drop off",
    "air conditioning",
    "non-smoking throughout",
    "non-smoking rooms",
    "dry cleaning/laundry",
    "private check-in",
    "express check-in",
    "business centre",
    "business center",
    "meeting/banquet",
    "currency exchange",
    "luggage storage",
    "private parking",
    "free self parking",
    "free parking",
    "airport shuttle",
    "shuttle service",
    "room service",
    "dry cleaning",
    "family rooms",
    "free wifi",
    "free wi-fi",
    "concierge",
    "restaurant",
    "breakfast",
    "elevator",
    "internet",
    "laundry",
    "parking",
    "smoking",
    "terrace",
    "garden",
    "heating",
    "fitness",
    "sauna",
    "wifi",
    "wi-fi",
    "pool",
    "gym",
    "spa",
    "pets",
    "bar",
)

FACILITY_KEYWORDS: tuple[str, ...] = tuple(sorted(_RAW_FACILITY_KEYWORDS, key=len, reverse=True))


def _facilities_plain(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    t = _strip_html_tags(s)
    return re.sub(r"\s+", " ", t).strip().lower()


def facilities_token_count(series: pd.Series) -> pd.Series:
    """Count facility tokens (rough verbosity / list length of ``HotelFacilities``)."""
    s = series.fillna("").astype(str).str.strip()
    empty = s.eq("") | s.str.lower().eq("nan")
    t = s.str.replace(r"<[^>]+>", " ", regex=True)
    t = t.str.replace(r"\s+", " ", regex=True).str.strip()
    t = t.str.replace(r"[,;/]+", " ", regex=True)
    counts = t.str.split().str.len().fillna(0).astype(int)
    out = pd.Series(counts, index=series.index, dtype="Int64")
    return out.where(~empty, 0).astype("Int64")


def _facilities_keyword_hits_one(raw: object) -> int:
    """How many curated amenity keywords/phrases appear (case-insensitive)."""
    low = _facilities_plain(raw)
    if not low:
        return 0
    low_noslash = low.replace("/", " ")
    n = 0
    for kw in FACILITY_KEYWORDS:
        if " " in kw or "/" in kw:
            if kw.replace("/", " ") in low_noslash:
                n += 1
        else:
            if re.search(rf"\b{re.escape(kw)}\b", low):
                n += 1
    return n


def facilities_keyword_hits(series: pd.Series) -> pd.Series:
    """Count matches against ``FACILITY_KEYWORDS`` in ``HotelFacilities`` (coarse amenity signal)."""
    s = series.fillna("").astype(str).str.strip()
    empty = s.eq("") | s.str.lower().eq("nan")
    counts = s.map(_facilities_keyword_hits_one)
    return counts.where(~empty, 0).astype("Int64")
