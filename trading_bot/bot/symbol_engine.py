from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StructureState:
    higher_highs: int
    higher_lows: int
    lower_highs: int
    lower_lows: int
    trend: str


@dataclass(slots=True)
class VolatilityState:
    bb_width: float
    bb_width_percentile: float
    atr_ratio: float
    atr_percentile: float
    range_tightening: bool
    energy_buildup: bool


@dataclass(slots=True)
class BreakoutState:
    direction: str
    level: float
    distance_from_level: float
    volume_expansion: float
    volatility_expansion: float
    label: str


@dataclass(slots=True)
class MomentumState:
    rsi_14: float
    ma50_distance_pct: float
    ma200_distance_pct: float
    profile: str


@dataclass(slots=True)
class RegimeState:
    regime: str
    move_classification: str
    event_like: bool


class SymbolSuggestionEngine:
    """Technical-only symbol ranking across stocks, crypto and commodities."""

    def __init__(
        self,
        *,
        breakout_lookback: int = 20,
        squeeze_percentile: float = 0.3,
        expansion_multiplier: float = 1.6,
    ) -> None:
        self.breakout_lookback = breakout_lookback
        self.squeeze_percentile = squeeze_percentile
        self.expansion_multiplier = expansion_multiplier

    def analyze_symbol(
        self,
        *,
        symbol: str,
        timeframe_data: dict[str, pd.DataFrame],
    ) -> dict:
        try:
            if not timeframe_data:
                raise ValueError("timeframe_data is empty")

            analyses: list[dict] = []
            for timeframe, candles in timeframe_data.items():
                analyses.append(self._analyze_timeframe(timeframe=timeframe, candles=candles))

            aggregate = self._aggregate_multi_timeframe(analyses)
            aggregate["symbol"] = symbol
            return aggregate
        except Exception as exc:
            logger.error(
                "symbol_engine analysis failed symbol=%s error=%s",
                symbol,
                exc,
            )
            raise

    def _analyze_timeframe(self, *, timeframe: str, candles: pd.DataFrame) -> dict:
        validated = self._validate_inputs(candles)
        structure = self._detect_market_structure(validated)
        volatility = self._volatility_compression(validated)
        breakout = self._detect_breakout(validated, structure=structure)
        momentum = self._momentum_extension(validated)
        regime = self._classify_regime(validated, structure=structure)
        score = self._score_symbol(
            structure=structure,
            breakout=breakout,
            volatility=volatility,
            momentum=momentum,
            regime=regime,
        )
        category = self._categorize(score=score, breakout=breakout, momentum=momentum)
        meta = self._meta_filter(score=score, breakout=breakout, regime=regime, momentum=momentum)

        return {
            "timeframe": timeframe,
            "structure": asdict(structure),
            "volatility": asdict(volatility),
            "breakout": asdict(breakout),
            "momentum": asdict(momentum),
            "regime": asdict(regime),
            "score": score,
            "category": category,
            "meta": meta,
        }

    def _validate_inputs(self, candles: pd.DataFrame) -> pd.DataFrame:
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(candles.columns)
        if missing:
            message = f"Missing required candle columns: {', '.join(sorted(missing))}"
            logger.error("symbol_engine invalid_input error=%s", message)
            raise ValueError(message)

        if len(candles) < 220:
            message = f"Need at least 220 candles, got {len(candles)}"
            logger.error("symbol_engine invalid_input error=%s", message)
            raise ValueError(message)

        return candles.astype(float).copy()

    def _detect_market_structure(self, candles: pd.DataFrame) -> StructureState:
        highs = candles["High"].tail(60).reset_index(drop=True)
        lows = candles["Low"].tail(60).reset_index(drop=True)

        hh = int((highs.diff() > 0).sum())
        hl = int((lows.diff() > 0).sum())
        lh = int((highs.diff() < 0).sum())
        ll = int((lows.diff() < 0).sum())

        if hh + hl > lh + ll + 12:
            trend = "bullish"
        elif lh + ll > hh + hl + 12:
            trend = "bearish"
        else:
            trend = "neutral"

        return StructureState(
            higher_highs=hh,
            higher_lows=hl,
            lower_highs=lh,
            lower_lows=ll,
            trend=trend,
        )

    def _volatility_compression(self, candles: pd.DataFrame) -> VolatilityState:
        close = candles["Close"]
        high = candles["High"]
        low = candles["Low"]

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std().fillna(0)
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)
        bb_width = ((bb_upper - bb_lower) / sma20.replace(0, pd.NA)).fillna(0)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14).mean().fillna(0)
        atr_ratio_series = (atr14 / close.replace(0, pd.NA)).fillna(0)

        bb_current = float(bb_width.iloc[-1])
        atr_current = float(atr_ratio_series.iloc[-1])
        bb_percentile = float((bb_width.rank(pct=True).iloc[-1]))
        atr_percentile = float((atr_ratio_series.rank(pct=True).iloc[-1]))

        rolling_range = (high.rolling(20).max() - low.rolling(20).min()).fillna(0)
        tightening = bool(rolling_range.iloc[-1] <= rolling_range.tail(10).median())

        energy_buildup = (
            bb_percentile <= self.squeeze_percentile
            and atr_percentile <= self.squeeze_percentile
            and tightening
        )

        return VolatilityState(
            bb_width=round(bb_current, 5),
            bb_width_percentile=round(bb_percentile, 4),
            atr_ratio=round(atr_current, 5),
            atr_percentile=round(atr_percentile, 4),
            range_tightening=tightening,
            energy_buildup=energy_buildup,
        )

    def _detect_breakout(
        self,
        candles: pd.DataFrame,
        *,
        structure: StructureState,
    ) -> BreakoutState:
        close = candles["Close"]
        high = candles["High"]
        low = candles["Low"]
        volume = candles["Volume"]

        lookback_high = float(high.iloc[-self.breakout_lookback - 1 : -1].max())
        lookback_low = float(low.iloc[-self.breakout_lookback - 1 : -1].min())
        latest_close = float(close.iloc[-1])

        vol_ratio = float(volume.iloc[-1] / max(volume.tail(21).mean(), 1e-9))
        true_range = (high - low).rolling(14).mean().fillna(0)
        volat_ratio = float(true_range.iloc[-1] / max(true_range.tail(21).mean(), 1e-9))

        if latest_close > lookback_high:
            direction = "up"
            level = lookback_high
            distance = ((latest_close / lookback_high) - 1) * 100
        elif latest_close < lookback_low:
            direction = "down"
            level = lookback_low
            distance = ((lookback_low / latest_close) - 1) * 100
        else:
            direction = "none"
            level = lookback_high if structure.trend == "bullish" else lookback_low
            distance = 0.0

        if direction == "none":
            label = "none"
        elif vol_ratio >= 2.0 and volat_ratio >= self.expansion_multiplier:
            label = "explosive breakout"
        elif vol_ratio >= 1.3 and volat_ratio >= 1.2:
            label = "confirmed breakout"
        else:
            label = "weak breakout"

        return BreakoutState(
            direction=direction,
            level=round(level, 4),
            distance_from_level=round(distance, 4),
            volume_expansion=round(vol_ratio, 4),
            volatility_expansion=round(volat_ratio, 4),
            label=label,
        )

    def _momentum_extension(self, candles: pd.DataFrame) -> MomentumState:
        close = candles["Close"]
        delta = close.diff().fillna(0)
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(14).mean().fillna(0)
        avg_loss = down.rolling(14).mean().fillna(0)
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        flat_mask = (avg_gain == 0) & (avg_loss == 0)
        rsi = rsi.where(~flat_mask, 50).fillna(50)

        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        current = close.iloc[-1]

        ma50_dist = ((current / ma50) - 1) * 100
        ma200_dist = ((current / ma200) - 1) * 100
        rsi_val = float(rsi.iloc[-1])

        if abs(ma50_dist) < 8 and abs(ma200_dist) < 18 and 45 <= rsi_val <= 70:
            profile = "healthy trend"
        elif abs(ma50_dist) > 16 or abs(ma200_dist) > 35 or rsi_val >= 80 or rsi_val <= 20:
            profile = "overextended (parabolic)"
        else:
            profile = "exhaustion risk"

        return MomentumState(
            rsi_14=round(rsi_val, 2),
            ma50_distance_pct=round(float(ma50_dist), 2),
            ma200_distance_pct=round(float(ma200_dist), 2),
            profile=profile,
        )

    def _classify_regime(self, candles: pd.DataFrame, *, structure: StructureState) -> RegimeState:
        close = candles["Close"]
        returns = close.pct_change().fillna(0)
        realized_vol = returns.rolling(20).std().fillna(0)
        current_vol = float(realized_vol.iloc[-1])
        vol_percentile = float(realized_vol.rank(pct=True).iloc[-1])

        last_move = float(abs(returns.iloc[-1]))
        move_to_vol = last_move / max(current_vol, 1e-9)
        event_like = move_to_vol > 2.8 and vol_percentile > 0.85 and last_move > 0.03

        trend_bias = (
            structure.higher_highs
            + structure.higher_lows
            - structure.lower_highs
            - structure.lower_lows
        )
        if vol_percentile < 0.35:
            regime = "accumulation"
        elif vol_percentile < 0.7:
            regime = "breakout"
        else:
            regime = "expansion"

        if event_like:
            classification = "event-driven move"
        elif abs(trend_bias) > 8 and move_to_vol < 2.0:
            classification = "technical trend"
        else:
            classification = "mixed"

        return RegimeState(regime=regime, move_classification=classification, event_like=event_like)

    def _score_symbol(
        self,
        *,
        structure: StructureState,
        breakout: BreakoutState,
        volatility: VolatilityState,
        momentum: MomentumState,
        regime: RegimeState,
    ) -> int:
        trend_points = 0
        trend_delta = (structure.higher_highs + structure.higher_lows) - (
            structure.lower_highs + structure.lower_lows
        )
        if structure.trend == "bullish":
            trend_points = min(max(8 + trend_delta // 2, 0), 25)
        elif structure.trend == "bearish":
            trend_points = min(max(8 + abs(trend_delta) // 2, 0), 25)

        breakout_points = {
            "none": 3,
            "weak breakout": 9,
            "confirmed breakout": 18,
            "explosive breakout": 24,
        }[breakout.label]

        volatility_points = 4
        if volatility.energy_buildup:
            volatility_points = 20
        elif volatility.range_tightening:
            volatility_points = 12

        momentum_points = {
            "healthy trend": 15,
            "exhaustion risk": 8,
            "overextended (parabolic)": 2,
        }[momentum.profile]

        risk_penalty = 0
        if momentum.profile == "overextended (parabolic)":
            risk_penalty -= 10
        if regime.move_classification == "event-driven move":
            risk_penalty -= 10

        raw_score = (
            trend_points + breakout_points + volatility_points + momentum_points + risk_penalty
        )
        return int(max(min(raw_score, 100), 0))

    def _categorize(self, *, score: int, breakout: BreakoutState, momentum: MomentumState) -> str:
        if breakout.label in {"confirmed breakout", "weak breakout"} and score >= 65:
            return "EARLY BREAKOUT"
        if score >= 58 and momentum.profile == "healthy trend":
            return "TREND CONTINUATION"
        if momentum.profile == "overextended (parabolic)" or score < 40:
            return "OVEREXTENDED (WAIT)"
        return "RANGE / NO EDGE"

    def _meta_filter(
        self,
        *,
        score: int,
        breakout: BreakoutState,
        regime: RegimeState,
        momentum: MomentumState,
    ) -> dict:
        should_take = True
        reason = "signal quality acceptable"

        if regime.event_like:
            should_take = False
            reason = "ignored: event-driven volatility spike"
        elif momentum.profile == "overextended (parabolic)":
            should_take = False
            reason = "ignored: extension risk too high"
        elif breakout.label == "weak breakout" and score < 62:
            should_take = False
            reason = "ignored: weak breakout without sufficient confluence"

        return {"decision": "take" if should_take else "ignore", "reason": reason}

    def _aggregate_multi_timeframe(self, analyses: list[dict]) -> dict:
        if not analyses:
            message = "No timeframe analyses generated"
            logger.error("symbol_engine aggregate error=%s", message)
            raise ValueError(message)

        avg_score = round(sum(item["score"] for item in analyses) / len(analyses), 2)

        priority_order = {
            "EARLY BREAKOUT": 4,
            "TREND CONTINUATION": 3,
            "RANGE / NO EDGE": 2,
            "OVEREXTENDED (WAIT)": 1,
        }
        dominant_category = max(
            analyses,
            key=lambda item: priority_order[item["category"]],
        )["category"]

        all_ignored = all(item["meta"]["decision"] == "ignore" for item in analyses)

        return {
            "overall_score": avg_score,
            "overall_category": "RANGE / NO EDGE" if all_ignored else dominant_category,
            "take_signal": not all_ignored,
            "timeframes": analyses,
        }


def rank_symbols(
    engine: SymbolSuggestionEngine,
    symbols_data: dict[str, dict[str, pd.DataFrame]],
) -> list[dict]:
    ranked: list[dict] = []
    for symbol, timeframe_data in symbols_data.items():
        ranked.append(engine.analyze_symbol(symbol=symbol, timeframe_data=timeframe_data))

    ranked.sort(
        key=lambda item: (
            item["take_signal"],
            item["overall_score"],
        ),
        reverse=True,
    )
    return ranked
