from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from trading_bot.bot.data_sources import fetch_trending_symbols, get_market_snapshot

logger = logging.getLogger(__name__)

EUROPE_TRENDING_REGIONS = (
    "IT", "FR", "GB", "AL", "AD", "AT", "BE", "BA", "BG", "BY", "CH", "CY",
    "CZ", "DK", "EE", "ES", "FI", "GR", "HR", "HU", "IE", "IS", "LI", "LT",
    "LU", "LV", "MC", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO",
    "RS", "RU", "SE", "SI", "SK", "SM", "TR", "UA", "VA",
)
TRENDING_REGIONS = ("US", *EUROPE_TRENDING_REGIONS)
DEFAULT_UNIVERSE_MAX_SYMBOLS = 60


def _parse_universe_regions() -> tuple[str, ...]:
    raw_regions = os.getenv("UNIVERSE_REGIONS", "").strip()
    if not raw_regions:
        return TRENDING_REGIONS

    regions = tuple(region.strip().upper() for region in raw_regions.split(",") if region.strip())
    if regions:
        return regions

    logger.error(
        "Invalid UNIVERSE_REGIONS=%s. Fallback to default TRENDING_REGIONS.",
        raw_regions,
    )
    return TRENDING_REGIONS


def _parse_universe_max_symbols() -> int:
    raw_value = os.getenv("UNIVERSE_MAX_SYMBOLS", str(DEFAULT_UNIVERSE_MAX_SYMBOLS)).strip()
    try:
        return max(int(raw_value), 1)
    except ValueError:
        logger.error(
            "Invalid UNIVERSE_MAX_SYMBOLS=%s. Fallback to %s.",
            raw_value,
            DEFAULT_UNIVERSE_MAX_SYMBOLS,
        )
        return DEFAULT_UNIVERSE_MAX_SYMBOLS


def build_candidate_universe(
    *,
    target_size: int,
    min_avg_volume_20d: float = 0.0,
    min_latest_close: float = 0.0,
    fetch_symbols_fn=fetch_trending_symbols,
    snapshot_fn=get_market_snapshot,
) -> list[dict[str, str]]:
    trending_provider = (
        os.getenv("MARKET_TRENDING_TICKERS_PROVIDER", "YFINANCE").strip().upper() or "YFINANCE"
    )
    universe_regions = ("GLOBAL",) if trending_provider == "ETORO" else _parse_universe_regions()
    universe_max_symbols = _parse_universe_max_symbols()
    fetch_limit = max(min(target_size * 2, universe_max_symbols), 1)

    symbols_by_region: dict[str, str] = {}
    for region in universe_regions:
        try:
            if trending_provider == "ETORO":
                region_symbols = fetch_symbols_fn(limit=fetch_limit)
            else:
                try:
                    region_symbols = fetch_symbols_fn(region=region, limit=fetch_limit)
                except TypeError:
                    region_symbols = fetch_symbols_fn(limit=fetch_limit)
        except RuntimeError as exc:
            logger.warning(
                "Skipping universe region=%s due to upstream error: %s",
                region,
                exc,
            )
            continue

        for symbol in region_symbols:
            normalized = str(symbol).strip().upper()
            if normalized:
                symbols_by_region.setdefault(normalized, region)
            if len(symbols_by_region) >= universe_max_symbols:
                break
        if len(symbols_by_region) >= universe_max_symbols:
            break

    if not symbols_by_region:
        error_message = (
            "Unable to fetch trending symbols from eToro."
            if trending_provider == "ETORO"
            else "Unable to fetch trending symbols from Yahoo Finance."
        )
        logger.error(
            "Candidate universe unavailable. provider=%s regions=%s universe_max_symbols=%s",
            trending_provider,
            ",".join(universe_regions),
            universe_max_symbols,
        )
        raise RuntimeError(error_message)

    filtered_symbols: list[dict[str, str]] = []
    for symbol, region in symbols_by_region.items():
        try:
            snapshot = snapshot_fn(symbol)
        except Exception as exc:
            logger.info("Skipping universe symbol=%s due to snapshot error=%s", symbol, exc)
            continue
        avg_volume_20d = float(getattr(snapshot, "avg_volume_20d", min_avg_volume_20d))
        latest_close = float(getattr(snapshot, "latest_close", min_latest_close))
        if avg_volume_20d < min_avg_volume_20d:
            continue
        if latest_close < min_latest_close:
            continue
        filtered_symbols.append({"symbol": symbol, "region": region})
        if len(filtered_symbols) >= universe_max_symbols:
            break

    if not filtered_symbols:
        logger.error(
            "Candidate universe empty after filters. min_avg_volume_20d=%s min_latest_close=%s",
            min_avg_volume_20d,
            min_latest_close,
        )
        raise RuntimeError("No symbols available after applying universe filters.")

    return filtered_symbols[: max(target_size, 1)]


def deduplicate_symbols(symbols: Iterable[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = str(symbol).strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(normalized)
    return deduplicated
