import json
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType


load_dotenv()


class ConfigError(Exception):
    """Raised when required configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    gamma_api_url: str
    clob_host: str
    chain_id: int
    private_key: str | None
    funder_address: str | None
    api_key: str | None
    api_secret: str | None
    api_passphrase: str | None
    poll_interval_seconds: int
    mispricing_threshold: Decimal
    exit_threshold: Decimal
    request_timeout_seconds: int
    max_markets_per_scan: int
    page_size: int
    max_active_positions: int
    max_total_exposure_usd: Decimal
    max_market_exposure_usd: Decimal
    max_loss_usd: Decimal
    take_profit_usd: Decimal
    stop_loss_usd: Decimal
    trade_usd_per_leg: Decimal
    min_order_size: Decimal
    price_buffer: Decimal
    log_level: str
    discord_webhook_url: str | None
    enable_trading: bool
    positions_file: str
    enable_volatility_sizing: bool
    volatility_lookback_days: int
    target_daily_volatility: Decimal
    min_volatility_multiplier: Decimal
    max_volatility_multiplier: Decimal
    volatility_update_seconds: int
    volatility_rolling_window_points: int
    volatility_source: str
    enable_dynamic_sizing: bool
    dynamic_size_exposure_pct: Decimal
    dynamic_size_min_usd_per_leg: Decimal
    dynamic_size_max_usd_per_leg: Decimal
    enable_pricing_diagnostics: bool
    diagnostics_top_n: int


@dataclass(frozen=True)
class MispricingEvent:
    market_id: str
    question: str
    yes_price: Decimal
    no_price: Decimal
    total: Decimal
    deviation: Decimal


@dataclass
class Position:
    market_id: str
    question: str
    side: str
    token_id: str
    quantity: Decimal
    average_price: Decimal


class PositionStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.positions: dict[str, Position] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for key, pos in raw.items():
                self.positions[key] = Position(
                    market_id=pos["market_id"],
                    question=pos["question"],
                    side=pos["side"],
                    token_id=pos["token_id"],
                    quantity=Decimal(str(pos["quantity"])),
                    average_price=Decimal(str(pos["average_price"])),
                )
        except Exception:
            logging.exception("Failed loading positions file %s; starting empty", self.path)
            self.positions = {}

    def save(self) -> None:
        serialized: dict[str, dict[str, Any]] = {}
        for key, pos in self.positions.items():
            data = asdict(pos)
            data["quantity"] = str(pos.quantity)
            data["average_price"] = str(pos.average_price)
            serialized[key] = data
        self.path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def key(self, market_id: str, side: str) -> str:
        return f"{market_id}:{side}"

    def upsert(self, pos: Position) -> None:
        self.positions[self.key(pos.market_id, pos.side)] = pos
        self.save()

    def get(self, market_id: str, side: str) -> Position | None:
        return self.positions.get(self.key(market_id, side))

    def remove(self, market_id: str, side: str) -> None:
        self.positions.pop(self.key(market_id, side), None)
        self.save()

    def count_active(self) -> int:
        return len(self.positions)

    def total_exposure(self) -> Decimal:
        return sum((p.quantity * p.average_price for p in self.positions.values()), Decimal("0"))

    def market_exposure(self, market_id: str) -> Decimal:
        return sum(
            (p.quantity * p.average_price for p in self.positions.values() if p.market_id == market_id),
            Decimal("0"),
        )

    def unrealized_pnl(self, market_prices: dict[str, dict[str, Decimal]]) -> Decimal:
        pnl = Decimal("0")
        for p in self.positions.values():
            m = market_prices.get(p.market_id)
            if not m:
                continue
            mark = m.get(p.side)
            if mark is None:
                continue
            pnl += (mark - p.average_price) * p.quantity
        return pnl


def configure_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ConfigError(f"Unsupported LOG_LEVEL: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got {value}")
    return value


def read_decimal_env(name: str, default: str) -> Decimal:
    raw = os.getenv(name, default)
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ConfigError(f"{name} must be a decimal number, got {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{name} must be >= 0, got {value}")
    return value


def read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true/false, got {raw!r}")


def load_settings() -> Settings:
    gamma_api_url = os.getenv("GAMMA_API_URL", "https://gamma-api.polymarket.com/markets")
    clob_host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
    if not gamma_api_url.lower().startswith(("http://", "https://")):
        raise ConfigError("GAMMA_API_URL must start with http:// or https://")
    if not clob_host.lower().startswith(("http://", "https://")):
        raise ConfigError("CLOB_HOST must start with http:// or https://")

    page_size = read_int_env("PAGE_SIZE", 200)
    max_markets_per_scan = read_int_env("MAX_MARKETS_PER_SCAN", 1000)
    if page_size > max_markets_per_scan:
        raise ConfigError("PAGE_SIZE cannot be greater than MAX_MARKETS_PER_SCAN")

    volatility_source = os.getenv("VOLATILITY_SOURCE", "coingecko").strip().lower()
    if volatility_source not in {"coingecko", "binance"}:
        raise ConfigError("VOLATILITY_SOURCE must be one of: coingecko, binance")

    enable_trading = read_bool_env("ENABLE_TRADING", False)
    dynamic_size_exposure_pct = read_decimal_env("DYNAMIC_SIZE_EXPOSURE_PCT", "10")
    if dynamic_size_exposure_pct > 100:
        raise ConfigError("DYNAMIC_SIZE_EXPOSURE_PCT must be <= 100")

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS")

    if enable_trading and not private_key:
        raise ConfigError("POLYMARKET_PRIVATE_KEY is required when ENABLE_TRADING=true")

    return Settings(
        gamma_api_url=gamma_api_url,
        clob_host=clob_host,
        chain_id=read_int_env("CHAIN_ID", 137),
        private_key=private_key,
        funder_address=funder_address,
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
        poll_interval_seconds=read_int_env("POLL_INTERVAL_SECONDS", 20),
        mispricing_threshold=read_decimal_env("MISPRICING_THRESHOLD", "0.03"),
        exit_threshold=read_decimal_env("EXIT_THRESHOLD", "0.01"),
        request_timeout_seconds=read_int_env("REQUEST_TIMEOUT_SECONDS", 15),
        max_markets_per_scan=max_markets_per_scan,
        page_size=page_size,
        max_active_positions=read_int_env("MAX_ACTIVE_POSITIONS", 20),
        max_total_exposure_usd=read_decimal_env("MAX_TOTAL_EXPOSURE_USD", "500"),
        max_market_exposure_usd=read_decimal_env("MAX_MARKET_EXPOSURE_USD", "100"),
        max_loss_usd=read_decimal_env("MAX_LOSS_USD", "100"),
        take_profit_usd=read_decimal_env("TAKE_PROFIT_USD", "15"),
        stop_loss_usd=read_decimal_env("STOP_LOSS_USD", "15"),
        trade_usd_per_leg=read_decimal_env("TRADE_USD_PER_LEG", "10"),
        min_order_size=read_decimal_env("MIN_ORDER_SIZE", "1"),
        price_buffer=read_decimal_env("PRICE_BUFFER", "0.01"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL") or None,
        enable_trading=enable_trading,
        positions_file=os.getenv("POSITIONS_FILE", "positions.json"),
        enable_volatility_sizing=read_bool_env("ENABLE_VOLATILITY_SIZING", False),
        volatility_lookback_days=read_int_env("VOLATILITY_LOOKBACK_DAYS", 14),
        target_daily_volatility=read_decimal_env("TARGET_DAILY_VOLATILITY", "0.03"),
        min_volatility_multiplier=read_decimal_env("MIN_VOLATILITY_MULTIPLIER", "0.25"),
        max_volatility_multiplier=read_decimal_env("MAX_VOLATILITY_MULTIPLIER", "1.50"),
        volatility_update_seconds=read_int_env("VOLATILITY_UPDATE_SECONDS", 60),
        volatility_rolling_window_points=read_int_env("VOLATILITY_ROLLING_WINDOW_POINTS", 120),
        volatility_source=volatility_source,
        enable_dynamic_sizing=read_bool_env("ENABLE_DYNAMIC_SIZING", False),
        dynamic_size_exposure_pct=dynamic_size_exposure_pct,
        dynamic_size_min_usd_per_leg=read_decimal_env("DYNAMIC_SIZE_MIN_USD_PER_LEG", "5"),
        dynamic_size_max_usd_per_leg=read_decimal_env("DYNAMIC_SIZE_MAX_USD_PER_LEG", "50"),
        enable_pricing_diagnostics=read_bool_env("ENABLE_PRICING_DIAGNOSTICS", True),
        diagnostics_top_n=read_int_env("DIAGNOSTICS_TOP_N", 10),
    )


def parse_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def extract_yes_no_prices(market: dict[str, Any]) -> tuple[Decimal | None, Decimal | None]:
    outcomes = parse_jsonish(market.get("outcomes"))
    outcome_prices = parse_jsonish(market.get("outcomePrices"))
    if isinstance(outcomes, list) and isinstance(outcome_prices, list):
        mapped: dict[str, Decimal] = {}
        for idx, label in enumerate(outcomes):
            if idx < len(outcome_prices) and isinstance(label, str):
                price = parse_decimal(outcome_prices[idx])
                if price is not None:
                    mapped[label.strip().upper()] = price
        return mapped.get("YES"), mapped.get("NO")
    return None, None


def extract_yes_no_token_ids(market: dict[str, Any]) -> tuple[str | None, str | None]:
    outcomes = parse_jsonish(market.get("outcomes"))
    token_ids = parse_jsonish(market.get("clobTokenIds"))
    if not (isinstance(outcomes, list) and isinstance(token_ids, list)):
        return None, None

    yes_token = None
    no_token = None
    for idx, label in enumerate(outcomes):
        if idx >= len(token_ids) or not isinstance(label, str):
            continue
        label_norm = label.strip().upper()
        tok = str(token_ids[idx])
        if label_norm == "YES":
            yes_token = tok
        elif label_norm == "NO":
            no_token = tok
    return yes_token, no_token


def fetch_markets_page(settings: Settings, offset: int) -> list[dict[str, Any]]:
    params = {"limit": settings.page_size, "offset": offset, "active": "true", "closed": "false"}
    response = requests.get(settings.gamma_api_url, params=params, timeout=settings.request_timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected API response: expected a list")
    return payload


def fetch_markets(settings: Settings) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    offset = 0
    while len(markets) < settings.max_markets_per_scan:
        page = fetch_markets_page(settings, offset)
        if not page:
            break
        remaining = settings.max_markets_per_scan - len(markets)
        markets.extend(page[:remaining])
        if len(page) < settings.page_size:
            break
        offset += settings.page_size
    return markets


def find_mispricings(markets: list[dict[str, Any]], threshold: Decimal) -> list[MispricingEvent]:
    events: list[MispricingEvent] = []
    for market in markets:
        yes_price, no_price = extract_yes_no_prices(market)
        if yes_price is None or no_price is None:
            continue
        total = yes_price + no_price
        deviation = abs(Decimal("1") - total)
        if deviation < threshold:
            continue
        events.append(
            MispricingEvent(
                market_id=str(market.get("id", "unknown")),
                question=str(market.get("question", "(no question)")),
                yes_price=yes_price,
                no_price=no_price,
                total=total,
                deviation=deviation,
            )
        )
    return events



def find_nearest_mispricings(markets: list[dict[str, Any]], threshold: Decimal, top_n: int) -> list[MispricingEvent]:
    candidates: list[MispricingEvent] = []
    for market in markets:
        yes_price, no_price = extract_yes_no_prices(market)
        if yes_price is None or no_price is None:
            continue
        total = yes_price + no_price
        deviation = abs(Decimal("1") - total)
        if deviation >= threshold:
            continue
        candidates.append(
            MispricingEvent(
                market_id=str(market.get("id", "unknown")),
                question=str(market.get("question", "(no question)")),
                yes_price=yes_price,
                no_price=no_price,
                total=total,
                deviation=deviation,
            )
        )

    candidates.sort(key=lambda e: (threshold - e.deviation, -e.deviation))
    return candidates[: max(1, top_n)]


def log_pricing_diagnostics(markets: list[dict[str, Any]], threshold: Decimal, top_n: int) -> None:
    nearest = find_nearest_mispricings(markets, threshold, top_n)
    if not nearest:
        logging.info("Diagnostics: no near-threshold markets found (top_n=%d)", top_n)
        return

    logging.info("Diagnostics: nearest %d markets below MISPRICING_THRESHOLD=%s", len(nearest), threshold)
    for idx, item in enumerate(nearest, start=1):
        gap = threshold - item.deviation
        logging.info(
            "Diag #%d | gap_to_trigger=%s | deviation=%s | total=%s | YES=%s NO=%s | ID=%s | %s",
            idx,
            gap,
            item.deviation,
            item.total,
            item.yes_price,
            item.no_price,
            item.market_id,
            item.question,
        )

def notify_discord(webhook_url: str, event: MispricingEvent, timeout_seconds: int) -> None:
    message = {
        "content": (
            "🚨 Polymarket mispricing detected\n"
            f"Market: {event.question}\n"
            f"ID: {event.market_id}\n"
            f"YES: {event.yes_price} | NO: {event.no_price}\n"
            f"YES+NO: {event.total} (deviation: {event.deviation})"
        )
    }
    response = requests.post(webhook_url, json=message, timeout=timeout_seconds)
    response.raise_for_status()


class Trader:
    def __init__(self, settings: Settings, store: PositionStore) -> None:
        self.settings = settings
        self.store = store
        self.client: ClobClient | None = None
        if settings.enable_trading:
            self.client = self._init_client()
        self._vol_cache: dict[str, tuple[float, Decimal]] = {}

    def _infer_underlying(self, question: str) -> str | None:
        q = question.upper()
        if "BTC" in q or "BITCOIN" in q:
            return "bitcoin"
        if "ETH" in q or "ETHEREUM" in q:
            return "ethereum"
        if "SOL" in q or "SOLANA" in q:
            return "solana"
        return None

    def _infer_binance_symbol(self, question: str) -> str | None:
        q = question.upper()
        if "BTC" in q or "BITCOIN" in q:
            return "BTCUSDT"
        if "ETH" in q or "ETHEREUM" in q:
            return "ETHUSDT"
        if "SOL" in q or "SOLANA" in q:
            return "SOLUSDT"
        return None

    def _fetch_binance_rolling_volatility(self, symbol: str) -> Decimal | None:
        now = time.time()
        cache_key = f"binance:{symbol}"
        cached = self._vol_cache.get(cache_key)
        if cached and now - cached[0] < self.settings.volatility_update_seconds:
            return cached[1]

        url = "https://api.binance.com/api/v3/klines"
        interval = "1m"
        limit = max(20, min(1000, self.settings.volatility_rolling_window_points + 5))
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            response = requests.get(url, params=params, timeout=self.settings.request_timeout_seconds)
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list) or len(rows) < 5:
                return None

            points: list[tuple[float, float]] = []
            for row in rows:
                if isinstance(row, list) and len(row) >= 5:
                    ts_ms = float(row[0])
                    close = float(row[4])
                    if close > 0:
                        points.append((ts_ms / 1000.0, close))

            if len(points) < 5:
                return None

            window = points[-self.settings.volatility_rolling_window_points :]
            returns: list[float] = []
            dt_seconds: list[float] = []
            for i in range(1, len(window)):
                t_prev, p_prev = window[i - 1]
                t_cur, p_cur = window[i]
                if p_prev <= 0 or p_cur <= 0 or t_cur <= t_prev:
                    continue
                returns.append(math.log(p_cur / p_prev))
                dt_seconds.append(t_cur - t_prev)

            if len(returns) < 4:
                return None

            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            avg_dt = sum(dt_seconds) / len(dt_seconds)
            if avg_dt <= 0:
                return None
            steps_per_day = 86400.0 / avg_dt
            daily_vol = math.sqrt(variance * steps_per_day)

            vol = Decimal(str(daily_vol))
            self._vol_cache[cache_key] = (now, vol)
            return vol
        except Exception:
            logging.exception("Failed fetching Binance rolling volatility for %s; using base size", symbol)
            return None

    def _fetch_rolling_volatility(self, coin_id: str) -> Decimal | None:
        now = time.time()
        cache_key = f"coingecko:{coin_id}"
        cached = self._vol_cache.get(cache_key)
        if cached and now - cached[0] < self.settings.volatility_update_seconds:
            return cached[1]

        # Pull intraday prices so volatility updates on a rolling basis.
        # CoinGecko updates are frequent and this endpoint returns short-horizon samples.
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "1",
            "interval": "",
        }

        try:
            response = requests.get(url, params=params, timeout=self.settings.request_timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            prices = payload.get("prices", [])

            points: list[tuple[float, float]] = []
            for row in prices:
                if isinstance(row, list) and len(row) >= 2:
                    ts_ms = float(row[0])
                    px = float(row[1])
                    if px > 0:
                        points.append((ts_ms / 1000.0, px))

            if len(points) < 5:
                return None

            window = points[-self.settings.volatility_rolling_window_points :]
            if len(window) < 5:
                return None

            returns: list[float] = []
            dt_seconds: list[float] = []
            for i in range(1, len(window)):
                t_prev, p_prev = window[i - 1]
                t_cur, p_cur = window[i]
                if p_prev <= 0 or p_cur <= 0 or t_cur <= t_prev:
                    continue
                returns.append(math.log(p_cur / p_prev))
                dt_seconds.append(t_cur - t_prev)

            if len(returns) < 4:
                return None

            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)

            # Convert per-step variance to daily realized volatility using average step duration.
            avg_dt = sum(dt_seconds) / len(dt_seconds)
            if avg_dt <= 0:
                return None
            steps_per_day = 86400.0 / avg_dt
            daily_vol = math.sqrt(variance * steps_per_day)

            vol = Decimal(str(daily_vol))
            self._vol_cache[cache_key] = (now, vol)
            return vol
        except Exception:
            logging.exception("Failed fetching rolling volatility for %s; using base size", coin_id)
            return None

    def _market_timeframe_scale(self, question: str) -> Decimal:
        q = question.lower()
        if any(k in q for k in ["today", "tonight", "hour", "by 5pm", "by 6pm", "by 8pm"]):
            return Decimal("1.25")
        if any(k in q for k in ["this week", "weekly", "by sunday", "by saturday"]):
            return Decimal("1.00")
        if any(k in q for k in ["this month", "monthly", "by month end", "by end of month"]):
            return Decimal("0.75")
        if any(k in q for k in ["this year", "year-end", "by december", "annual", "2026", "2027", "2028"]):
            return Decimal("0.60")
        return Decimal("1.00")

    def _event_end_timestamp(self, market: dict[str, Any]) -> datetime | None:
        for key in ("endDate", "end_date", "closeTime", "close_time", "eventEndDate", "event_end_date"):
            value = market.get(key)
            if not value:
                continue
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                continue
        return None

    def _time_to_event_hours(self, market: dict[str, Any]) -> float | None:
        end_ts = self._event_end_timestamp(market)
        if not end_ts:
            return None
        now = datetime.now(timezone.utc)
        delta_hours = (end_ts - now).total_seconds() / 3600.0
        return max(0.0, delta_hours)

    def _volatility_multiplier(self, question: str, market: dict[str, Any] | None = None) -> Decimal:
        if not self.settings.enable_volatility_sizing:
            return Decimal("1")

        coin_id = self._infer_underlying(question)
        if not coin_id:
            return Decimal("1")

        realized_vol: Decimal | None
        if self.settings.volatility_source == "binance":
            symbol = self._infer_binance_symbol(question)
            if not symbol:
                return Decimal("1")
            realized_vol = self._fetch_binance_rolling_volatility(symbol)
        else:
            realized_vol = self._fetch_rolling_volatility(coin_id)

        if not realized_vol or realized_vol <= 0:
            return Decimal("1")

        target = self.settings.target_daily_volatility
        timeframe_scale = self._market_timeframe_scale(question)
        if market is not None:
            hours_to_event = self._time_to_event_hours(market)
            if hours_to_event is not None:
                if hours_to_event <= 6:
                    timeframe_scale *= Decimal("0.55")
                elif hours_to_event <= 24:
                    timeframe_scale *= Decimal("0.75")
                elif hours_to_event <= 72:
                    timeframe_scale *= Decimal("0.90")
                elif hours_to_event >= 24 * 14:
                    timeframe_scale *= Decimal("1.10")

        raw = (target / realized_vol) * timeframe_scale
        return max(self.settings.min_volatility_multiplier, min(self.settings.max_volatility_multiplier, raw))

    def _init_client(self) -> ClobClient:
        if not self.settings.private_key:
            raise ConfigError("Trading enabled but private key missing")

        client = ClobClient(
            host=self.settings.clob_host,
            key=self.settings.private_key,
            chain_id=self.settings.chain_id,
            signature_type=2,
            funder=self.settings.funder_address,
        )

        if self.settings.api_key and self.settings.api_secret and self.settings.api_passphrase:
            creds = ApiCreds(
                api_key=self.settings.api_key,
                api_secret=self.settings.api_secret,
                api_passphrase=self.settings.api_passphrase,
            )
            client.set_api_creds(creds)
        else:
            client.set_api_creds(client.create_or_derive_api_creds())

        logging.info("Trading enabled: connected to CLOB host %s", self.settings.clob_host)
        return client

    def _risk_allows_new_trade(
        self,
        market_id: str,
        market_prices: dict[str, dict[str, Decimal]],
        intended_usd_per_leg: Decimal,
    ) -> bool:
        if self.store.count_active() >= self.settings.max_active_positions:
            logging.warning("Risk check blocked trade: MAX_ACTIVE_POSITIONS reached")
            return False

        if self.store.total_exposure() + (intended_usd_per_leg * 2) > self.settings.max_total_exposure_usd:
            logging.warning("Risk check blocked trade: MAX_TOTAL_EXPOSURE_USD reached")
            return False

        if self.store.market_exposure(market_id) + (intended_usd_per_leg * 2) > self.settings.max_market_exposure_usd:
            logging.warning("Risk check blocked trade: MAX_MARKET_EXPOSURE_USD reached for market %s", market_id)
            return False

        pnl = self.store.unrealized_pnl(market_prices)
        if pnl <= -self.settings.max_loss_usd:
            logging.warning("Risk check blocked trade: max loss hit (unrealized pnl=%s)", pnl)
            return False

        return True

    def _place_order(self, token_id: str, side: str, price: Decimal, size: Decimal) -> None:
        if self.client is None:
            return
        order = OrderArgs(
            token_id=token_id,
            price=float(price),
            size=float(size),
            side=side,
        )
        signed_order = self.client.create_order(order)
        self.client.post_order(signed_order, OrderType.GTC)

    def _calc_size(self, price: Decimal, usd_notional: Decimal) -> Decimal:
        if price <= 0:
            return Decimal("0")
        return (usd_notional / price).quantize(Decimal("0.0001"))

    def _intended_usd_per_leg(self, event: MispricingEvent, vol_multiplier: Decimal) -> Decimal:
        base = self.settings.trade_usd_per_leg * vol_multiplier
        if not self.settings.enable_dynamic_sizing:
            return base.quantize(Decimal("0.0001"))

        threshold = self.settings.mispricing_threshold or Decimal("0.0001")
        signal_factor = event.deviation / threshold
        signal_factor = max(Decimal("0.5"), min(Decimal("2.0"), signal_factor))

        max_total = self.settings.max_total_exposure_usd
        if max_total <= 0:
            headroom_factor = Decimal("1")
        else:
            remaining = max(Decimal("0"), max_total - self.store.total_exposure())
            headroom_factor = remaining / max_total
            headroom_factor = max(Decimal("0.25"), min(Decimal("1.0"), headroom_factor))

        pct_cap_per_leg = (self.settings.max_total_exposure_usd * (self.settings.dynamic_size_exposure_pct / Decimal("100"))) / Decimal("2")
        dynamic = base * signal_factor * headroom_factor
        dynamic = min(dynamic, pct_cap_per_leg)
        dynamic = max(self.settings.dynamic_size_min_usd_per_leg, dynamic)
        dynamic = min(self.settings.dynamic_size_max_usd_per_leg, dynamic)
        return dynamic.quantize(Decimal("0.0001"))

    def enter_if_signal(self, market: dict[str, Any], event: MispricingEvent, market_prices: dict[str, dict[str, Decimal]]) -> None:
        if not self.settings.enable_trading:
            return
        if event.total >= (Decimal("1") - self.settings.mispricing_threshold):
            return

        yes_token, no_token = extract_yes_no_token_ids(market)
        if not yes_token or not no_token:
            logging.warning("Skipping trade: missing token IDs for market %s", event.market_id)
            return

        if self.store.get(event.market_id, "YES") or self.store.get(event.market_id, "NO"):
            return

        vol_multiplier = self._volatility_multiplier(event.question, market)
        intended_usd_per_leg = self._intended_usd_per_leg(event, vol_multiplier)

        if not self._risk_allows_new_trade(event.market_id, market_prices, intended_usd_per_leg):
            return

        yes_buy_price = min(Decimal("0.99"), event.yes_price + self.settings.price_buffer)
        no_buy_price = min(Decimal("0.99"), event.no_price + self.settings.price_buffer)
        yes_size = self._calc_size(yes_buy_price, intended_usd_per_leg)
        no_size = self._calc_size(no_buy_price, intended_usd_per_leg)

        if yes_size < self.settings.min_order_size or no_size < self.settings.min_order_size:
            logging.warning("Skipping trade: calculated size below MIN_ORDER_SIZE")
            return

        try:
            self._place_order(yes_token, "BUY", yes_buy_price, yes_size)
            self._place_order(no_token, "BUY", no_buy_price, no_size)

            self.store.upsert(
                Position(
                    market_id=event.market_id,
                    question=event.question,
                    side="YES",
                    token_id=yes_token,
                    quantity=yes_size,
                    average_price=yes_buy_price,
                )
            )
            self.store.upsert(
                Position(
                    market_id=event.market_id,
                    question=event.question,
                    side="NO",
                    token_id=no_token,
                    quantity=no_size,
                    average_price=no_buy_price,
                )
            )
            logging.info(
                "Entered market %s with YES and NO legs (usd_per_leg=%s, vol_multiplier=%s, dynamic_sizing=%s)",
                event.market_id,
                intended_usd_per_leg,
                vol_multiplier,
                self.settings.enable_dynamic_sizing,
            )
        except Exception:
            logging.exception("Failed executing entry orders for market %s", event.market_id)

    def _market_unrealized_pnl(self, market_id: str, market_prices: dict[str, dict[str, Decimal]]) -> Decimal:
        pnl = Decimal("0")
        prices = market_prices.get(market_id, {})
        for side in ("YES", "NO"):
            pos = self.store.get(market_id, side)
            if not pos:
                continue
            mark = prices.get(side)
            if mark is None:
                continue
            pnl += (mark - pos.average_price) * pos.quantity
        return pnl

    def _close_market_positions(self, market_id: str, prices: dict[str, Decimal], reason: str) -> bool:
        yes_pos = self.store.get(market_id, "YES")
        no_pos = self.store.get(market_id, "NO")
        if not yes_pos and not no_pos:
            return False

        try:
            if yes_pos:
                yes_mark = prices.get("YES")
                if yes_mark is not None:
                    sell_price = max(Decimal("0.01"), yes_mark - self.settings.price_buffer)
                    self._place_order(yes_pos.token_id, "SELL", sell_price, yes_pos.quantity)
                self.store.remove(market_id, "YES")

            if no_pos:
                no_mark = prices.get("NO")
                if no_mark is not None:
                    sell_price = max(Decimal("0.01"), no_mark - self.settings.price_buffer)
                    self._place_order(no_pos.token_id, "SELL", sell_price, no_pos.quantity)
                self.store.remove(market_id, "NO")

            logging.info("Exited positions for market %s (reason=%s)", market_id, reason)
            return True
        except Exception:
            logging.exception("Failed executing exits for market %s (reason=%s)", market_id, reason)
            return False

    def enforce_risk_exits(self, market_prices: dict[str, dict[str, Decimal]]) -> None:
        if not self.settings.enable_trading:
            return

        active_market_ids = {pos.market_id for pos in self.store.positions.values()}
        for market_id in active_market_ids:
            prices = market_prices.get(market_id)
            if not prices:
                continue
            market_pnl = self._market_unrealized_pnl(market_id, market_prices)
            if market_pnl >= self.settings.take_profit_usd:
                self._close_market_positions(market_id, prices, "take_profit")
            elif market_pnl <= -self.settings.stop_loss_usd:
                self._close_market_positions(market_id, prices, "stop_loss")

    def exit_if_signal(self, market: dict[str, Any], event: MispricingEvent) -> None:
        if not self.settings.enable_trading:
            return

        if abs(Decimal("1") - event.total) > self.settings.exit_threshold:
            return

        prices = {"YES": event.yes_price, "NO": event.no_price}
        self._close_market_positions(event.market_id, prices, "normalization")



def run_bot(settings: Settings) -> None:
    should_stop = False
    seen_market_ids: set[str] = set()
    store = PositionStore(settings.positions_file)
    trader = Trader(settings, store)

    def _handle_stop_signal(signum: int, _frame: Any) -> None:
        nonlocal should_stop
        should_stop = True
        logging.info("Received signal %s, shutting down gracefully...", signum)

    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    logging.info("Starting Polymarket mispricing bot")
    logging.info("Polling %s every %ss", settings.gamma_api_url, settings.poll_interval_seconds)

    while not should_stop:
        started = time.time()
        try:
            markets = fetch_markets(settings)
            market_by_id = {str(m.get("id", "")): m for m in markets if m.get("id") is not None}
            current_ids = set(market_by_id.keys())
            new_market_count = len(current_ids - seen_market_ids)
            seen_market_ids.update(current_ids)

            market_prices: dict[str, dict[str, Decimal]] = {}
            for market in markets:
                market_id = str(market.get("id", ""))
                yes_p, no_p = extract_yes_no_prices(market)
                if yes_p is not None and no_p is not None:
                    market_prices[market_id] = {"YES": yes_p, "NO": no_p}

            logging.info(
                "Scanned %d active markets (%d newly observed this run)",
                len(markets),
                new_market_count,
            )

            events = find_mispricings(markets, settings.mispricing_threshold)
            if not events:
                logging.info("No mispricings above threshold found in this cycle")
                if settings.enable_pricing_diagnostics:
                    log_pricing_diagnostics(markets, settings.mispricing_threshold, settings.diagnostics_top_n)

            trader.enforce_risk_exits(market_prices)

            for event in events:
                logging.warning(
                    "Mispricing | ID=%s | deviation=%s | YES=%s NO=%s | %s",
                    event.market_id,
                    event.deviation,
                    event.yes_price,
                    event.no_price,
                    event.question,
                )
                if settings.discord_webhook_url:
                    try:
                        notify_discord(settings.discord_webhook_url, event, settings.request_timeout_seconds)
                    except requests.RequestException:
                        logging.exception("Failed to send Discord alert for market %s", event.market_id)

                market = market_by_id.get(event.market_id)
                if market:
                    trader.enter_if_signal(market, event, market_prices)
                    trader.exit_if_signal(market, event)

        except requests.RequestException:
            logging.exception("HTTP/API error while fetching market data")
        except Exception:
            logging.exception("Unexpected runtime error in bot loop")

        elapsed = time.time() - started
        sleep_time = max(0, settings.poll_interval_seconds - elapsed)
        if sleep_time > 0 and not should_stop:
            time.sleep(sleep_time)

    logging.info("Bot stopped")


def main() -> int:
    try:
        settings = load_settings()
        configure_logging(settings.log_level)
        run_bot(settings)
    except ConfigError as exc:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(message)s")
        logging.error("Configuration error: %s", exc)
        return 2
    except Exception:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(message)s")
        logging.exception("Fatal startup error")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
