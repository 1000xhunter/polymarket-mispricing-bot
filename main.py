import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests
import websocket
from websocket import WebSocketTimeoutException
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType

load_dotenv()


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    gamma_api_url: str
    clob_host: str
    clob_ws_url: str
    request_timeout_seconds: int
    poll_interval_seconds: int
    eval_interval_seconds: int
    max_markets_per_scan: int
    page_size: int
    log_level: str

    crypto_only: bool
    crypto_keywords: tuple[str, ...]
    use_midpoint_snapshot: bool
    alert_cooldown_seconds: int
    enable_diagnostics: bool
    diagnostics_top_n: int

    # Pinch / CDF thresholds
    pinch_score_threshold: Decimal
    narrow_band_prob_threshold: Decimal
    narrow_band_width_pct_threshold: Decimal
    monotonicity_tolerance: Decimal
    correction_drop_pct: Decimal
    correction_pinch_below: Decimal

    # Volatility + time-to-event
    enable_volatility_context: bool
    volatility_source: str
    volatility_update_seconds: int
    volatility_rolling_window_points: int

    # Alerts
    enable_discord_alerts: bool
    discord_webhook_url: str | None

    # Execution
    enable_trading: bool
    enable_paper_execution: bool
    chain_id: int
    private_key: str | None
    funder_address: str | None
    api_key: str | None
    api_secret: str | None
    api_passphrase: str | None
    trade_usd_per_leg: Decimal
    min_order_size: Decimal
    price_buffer: Decimal
    max_open_spreads: int
    positions_file: str
    ws_recv_timeout_seconds: int
    ws_heartbeat_seconds: int
    ws_reconnect_seconds: int
    clear_console_seconds: int
    dashboard_interval_seconds: int
    enable_windows_title: bool


def read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true/false, got {raw!r}")


def read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be int, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be > 0")
    return value


def read_decimal_env(name: str, default: str) -> Decimal:
    raw = os.getenv(name, default)
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ConfigError(f"{name} must be decimal, got {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{name} must be >= 0")
    return value


def configure_logging(log_level: str) -> None:
    numeric = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric, int):
        raise ConfigError(f"Unsupported LOG_LEVEL: {log_level}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_settings() -> Settings:
    gamma_api_url = os.getenv("GAMMA_API_URL", "https://gamma-api.polymarket.com/markets")
    clob_host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
    clob_ws_url = os.getenv("CLOB_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market")

    if not gamma_api_url.startswith(("http://", "https://")):
        raise ConfigError("GAMMA_API_URL must start with http:// or https://")
    if not clob_host.startswith(("http://", "https://")):
        raise ConfigError("CLOB_HOST must start with http:// or https://")
    if not clob_ws_url.startswith(("ws://", "wss://")):
        raise ConfigError("CLOB_WS_URL must start with ws:// or wss://")

    crypto_keywords_raw = os.getenv("CRYPTO_KEYWORDS", "btc,bitcoin,eth,ethereum,sol,solana,crypto,doge,xrp")
    crypto_keywords = tuple(k.strip().lower() for k in crypto_keywords_raw.split(",") if k.strip())
    if not crypto_keywords:
        raise ConfigError("CRYPTO_KEYWORDS must not be empty")

    volatility_source = os.getenv("VOLATILITY_SOURCE", "binance").strip().lower()
    if volatility_source not in {"binance", "coingecko"}:
        raise ConfigError("VOLATILITY_SOURCE must be binance or coingecko")

    clear_console_raw = os.getenv("CLEAR_CONSOLE_SECONDS", "60")
    try:
        clear_console_seconds = int(clear_console_raw)
    except ValueError as exc:
        raise ConfigError(f"CLEAR_CONSOLE_SECONDS must be int, got {clear_console_raw!r}") from exc
    if clear_console_seconds < 0:
        raise ConfigError("CLEAR_CONSOLE_SECONDS must be >= 0")

    enable_trading = read_bool_env("ENABLE_TRADING", False)
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if enable_trading and not private_key:
        raise ConfigError("POLYMARKET_PRIVATE_KEY required when ENABLE_TRADING=true")

    return Settings(
        gamma_api_url=gamma_api_url,
        clob_host=clob_host,
        clob_ws_url=clob_ws_url,
        request_timeout_seconds=read_int_env("REQUEST_TIMEOUT_SECONDS", 15),
        poll_interval_seconds=read_int_env("POLL_INTERVAL_SECONDS", 60),
        eval_interval_seconds=read_int_env("EVAL_INTERVAL_SECONDS", 2),
        max_markets_per_scan=read_int_env("MAX_MARKETS_PER_SCAN", 2000),
        page_size=read_int_env("PAGE_SIZE", 200),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        crypto_only=read_bool_env("CRYPTO_ONLY", True),
        crypto_keywords=crypto_keywords,
        use_midpoint_snapshot=read_bool_env("USE_MIDPOINT_SNAPSHOT", True),
        alert_cooldown_seconds=read_int_env("ALERT_COOLDOWN_SECONDS", 30),
        enable_diagnostics=read_bool_env("ENABLE_DIAGNOSTICS", True),
        diagnostics_top_n=read_int_env("DIAGNOSTICS_TOP_N", 20),
        pinch_score_threshold=read_decimal_env("PINCH_SCORE_THRESHOLD", "0.8"),
        narrow_band_prob_threshold=read_decimal_env("NARROW_BAND_PROB_THRESHOLD", "0.35"),
        narrow_band_width_pct_threshold=read_decimal_env("NARROW_BAND_WIDTH_PCT_THRESHOLD", "0.005"),
        monotonicity_tolerance=read_decimal_env("MONOTONICITY_TOLERANCE", "0.01"),
        correction_drop_pct=read_decimal_env("CORRECTION_DROP_PCT", "0.30"),
        correction_pinch_below=read_decimal_env("CORRECTION_PINCH_BELOW", "0.4"),
        enable_volatility_context=read_bool_env("ENABLE_VOLATILITY_CONTEXT", True),
        volatility_source=volatility_source,
        volatility_update_seconds=read_int_env("VOLATILITY_UPDATE_SECONDS", 60),
        volatility_rolling_window_points=read_int_env("VOLATILITY_ROLLING_WINDOW_POINTS", 120),
        enable_discord_alerts=read_bool_env("ENABLE_DISCORD_ALERTS", False),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL") or None,
        enable_trading=enable_trading,
        enable_paper_execution=read_bool_env("ENABLE_PAPER_EXECUTION", True),
        chain_id=read_int_env("CHAIN_ID", 137),
        private_key=private_key,
        funder_address=os.getenv("POLYMARKET_FUNDER_ADDRESS") or None,
        api_key=os.getenv("POLYMARKET_API_KEY") or None,
        api_secret=os.getenv("POLYMARKET_API_SECRET") or None,
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE") or None,
        trade_usd_per_leg=read_decimal_env("TRADE_USD_PER_LEG", "10"),
        min_order_size=read_decimal_env("MIN_ORDER_SIZE", "1"),
        price_buffer=read_decimal_env("PRICE_BUFFER", "0.01"),
        max_open_spreads=read_int_env("MAX_OPEN_SPREADS", 30),
        positions_file=os.getenv("POSITIONS_FILE", "spread_positions.json"),
        ws_recv_timeout_seconds=read_int_env("WS_RECV_TIMEOUT_SECONDS", 20),
        ws_heartbeat_seconds=read_int_env("WS_HEARTBEAT_SECONDS", 15),
        ws_reconnect_seconds=read_int_env("WS_RECONNECT_SECONDS", 3),
        clear_console_seconds=clear_console_seconds,
        dashboard_interval_seconds=read_int_env("DASHBOARD_INTERVAL_SECONDS", 10),
        enable_windows_title=read_bool_env("ENABLE_WINDOWS_TITLE", True),
    )


def set_console_title(title: str, settings: Settings) -> None:
    if not settings.enable_windows_title:
        return
    try:
        if os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.SetConsoleTitleW(title)
        elif sys.stdout.isatty():
            print(f"\33]0;{title}\a", end="", flush=True)
    except Exception:
        logging.debug("Unable to set console title", exc_info=True)


def log_cycle_dashboard(
    settings: Settings,
    cycle: int,
    markets_count: int,
    asset_count: int,
    signal_count: int,
    open_spreads: int,
    realized: Decimal,
    unrealized: Decimal,
    total: Decimal,
) -> None:
    mode = "LIVE" if settings.enable_trading else "PAPER"
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    logging.info("=" * 72)
    logging.info(
        "Dashboard | cycle=%d | %s | mode=%s | markets=%d | assets=%d | signals=%d | open_spreads=%d",
        cycle,
        now_utc,
        mode,
        markets_count,
        asset_count,
        signal_count,
        open_spreads,
    )
    logging.info(
        "P&L      | realized=%s | unrealized=%s | total=%s",
        realized,
        unrealized,
        total,
    )
    logging.info("Hints    | Ctrl+C to stop | ENABLE_TRADING=%s | diagnostics=%s", settings.enable_trading, settings.enable_diagnostics)
    logging.info("=" * 72)
    set_console_title(
        f"Polymarket Bot | {mode} | signals={signal_count} | open={open_spreads} | pnl={total}",
        settings,
    )


@dataclass(frozen=True)
class ThresholdMarket:
    market_id: str
    event_name: str
    question: str
    underlying: str
    direction: str
    level: Decimal
    yes_asset_id: str
    condition_id: str | None
    end_ts: datetime | None


@dataclass
class BookState:
    bid: Decimal | None = None
    ask: Decimal | None = None
    mid: Decimal | None = None
    updated_at: float = 0.0


@dataclass
class PinchSignal:
    event_key: str
    event_name: str
    underlying: str
    level_low: Decimal
    level_high: Decimal
    low_asset_id: str
    high_asset_id: str
    p_low: Decimal
    p_high: Decimal
    band_prob: Decimal
    band_width_pct: Decimal
    pinch: Decimal
    reason: str
    suggestion: str
    hours_to_event: float | None
    duration_scaled_vol: Decimal | None


@dataclass
class SpreadPosition:
    key: str
    event_key: str
    low_asset_id: str
    high_asset_id: str
    low_side: str
    high_side: str
    low_qty: Decimal
    high_qty: Decimal
    low_entry_price: Decimal
    high_entry_price: Decimal
    entry_band_prob: Decimal
    entry_pinch: Decimal
    opened_at: float


class SpreadStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.positions: dict[str, SpreadPosition] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for k, p in raw.items():
                self.positions[k] = SpreadPosition(
                    key=k,
                    event_key=p["event_key"],
                    low_asset_id=p["low_asset_id"],
                    high_asset_id=p["high_asset_id"],
                    low_side=p["low_side"],
                    high_side=p["high_side"],
                    low_qty=Decimal(str(p["low_qty"])),
                    high_qty=Decimal(str(p["high_qty"])),
                    low_entry_price=Decimal(str(p["low_entry_price"])),
                    high_entry_price=Decimal(str(p["high_entry_price"])),
                    entry_band_prob=Decimal(str(p["entry_band_prob"])),
                    entry_pinch=Decimal(str(p["entry_pinch"])),
                    opened_at=float(p["opened_at"]),
                )
        except Exception:
            logging.exception("Failed loading spread positions")
            self.positions = {}

    def _save(self) -> None:
        serialized: dict[str, Any] = {}
        for k, p in self.positions.items():
            d = asdict(p)
            for fld in ("low_qty", "high_qty", "low_entry_price", "high_entry_price", "entry_band_prob", "entry_pinch"):
                d[fld] = str(d[fld])
            serialized[k] = d
        self.path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def upsert(self, pos: SpreadPosition) -> None:
        self.positions[pos.key] = pos
        self._save()

    def remove(self, key: str) -> None:
        self.positions.pop(key, None)
        self._save()


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


def parse_end_timestamp(market: dict[str, Any]) -> datetime | None:
    for key in ("endDate", "end_date", "closeTime", "close_time", "eventEndDate", "event_end_date"):
        raw = market.get(key)
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


THRESHOLD_PATTERNS = [
    re.compile(r"\b(?P<asset>BTC|BITCOIN|ETH|ETHEREUM|SOL|SOLANA)\b.*?\b(above|over|at least)\s*\$?(?P<level>\d+(?:[\.,]\d+)?)", re.IGNORECASE),
    re.compile(r"\b(?P<asset>BTC|BITCOIN|ETH|ETHEREUM|SOL|SOLANA)\b.*?\b(below|under|at most)\s*\$?(?P<level>\d+(?:[\.,]\d+)?)", re.IGNORECASE),
]


def parse_threshold_from_question(question: str) -> tuple[str, str, Decimal] | None:
    for pattern in THRESHOLD_PATTERNS:
        m = pattern.search(question.strip())
        if not m:
            continue
        asset = m.group("asset").upper()
        level = parse_decimal(m.group("level").replace(",", ""))
        if level is None:
            continue
        token = m.group(2).lower()
        direction = "above" if token in {"above", "over", "at least"} else "below"
        return asset, direction, level
    return None


def extract_yes_asset_id(market: dict[str, Any]) -> str | None:
    outcomes = parse_jsonish(market.get("outcomes"))
    token_ids = parse_jsonish(market.get("clobTokenIds"))
    if not (isinstance(outcomes, list) and isinstance(token_ids, list)):
        return None
    for idx, label in enumerate(outcomes):
        if idx < len(token_ids) and isinstance(label, str) and label.strip().upper() == "YES":
            return str(token_ids[idx])
    return None


def _market_text_blob(market: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("question", "description", "slug", "category", "title", "eventTitle", "event_title"):
        value = market.get(key)
        if value:
            parts.append(str(value))
    tags = market.get("tags")
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags)
    elif tags:
        parts.append(str(tags))
    return " ".join(parts).lower()


def fetch_markets(settings: Settings) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    offset = 0
    while len(markets) < settings.max_markets_per_scan:
        params = {"active": "true", "closed": "false", "limit": settings.page_size, "offset": offset}
        resp = requests.get(settings.gamma_api_url, params=params, timeout=settings.request_timeout_seconds)
        resp.raise_for_status()
        page = resp.json()
        if not isinstance(page, list) or not page:
            break
        remaining = settings.max_markets_per_scan - len(markets)
        markets.extend(page[:remaining])
        if len(page) < settings.page_size:
            break
        offset += settings.page_size
    return markets


def discover_threshold_markets(settings: Settings) -> list[ThresholdMarket]:
    raw = fetch_markets(settings)
    if settings.crypto_only:
        raw = [m for m in raw if any(k in _market_text_blob(m) for k in settings.crypto_keywords)]

    out: list[ThresholdMarket] = []
    for m in raw:
        q = str(m.get("question", ""))
        parsed = parse_threshold_from_question(q)
        if not parsed:
            continue
        yes_id = extract_yes_asset_id(m)
        if not yes_id:
            continue
        asset, direction, level = parsed
        event_name = str(m.get("eventTitle") or m.get("title") or q)
        cond = str(m.get("conditionId")) if m.get("conditionId") else None
        out.append(
            ThresholdMarket(
                market_id=str(m.get("id", "")),
                event_name=event_name,
                question=q,
                underlying=asset,
                direction=direction,
                level=level,
                yes_asset_id=yes_id,
                condition_id=cond,
                end_ts=parse_end_timestamp(m),
            )
        )
    logging.info("Discovered %d threshold-style crypto markets", len(out))
    return out


class BookTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._books: dict[str, BookState] = {}

    def upsert(self, asset_id: str, bid: Decimal | None, ask: Decimal | None, mid: Decimal | None) -> None:
        with self._lock:
            st = self._books.get(asset_id, BookState())
            if bid is not None:
                st.bid = bid
            if ask is not None:
                st.ask = ask
            if mid is not None:
                st.mid = mid
            elif st.bid is not None and st.ask is not None:
                st.mid = (st.bid + st.ask) / Decimal("2")
            st.updated_at = time.time()
            self._books[asset_id] = st

    def get_mid(self, asset_id: str) -> Decimal | None:
        with self._lock:
            st = self._books.get(asset_id)
            if not st:
                return None
            if st.mid is not None:
                return st.mid
            if st.bid is not None and st.ask is not None:
                return (st.bid + st.ask) / Decimal("2")
            return None


def parse_book_message(payload: dict[str, Any]) -> list[tuple[str, Decimal | None, Decimal | None, Decimal | None]]:
    updates: list[tuple[str, Decimal | None, Decimal | None, Decimal | None]] = []

    def process_obj(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        asset_id = obj.get("asset_id") or obj.get("assetId") or obj.get("token_id") or obj.get("market")
        bid = parse_decimal(obj.get("best_bid") or obj.get("bid") or obj.get("b"))
        ask = parse_decimal(obj.get("best_ask") or obj.get("ask") or obj.get("a"))
        mid = parse_decimal(obj.get("mid") or obj.get("midpoint") or obj.get("price"))

        bids = obj.get("bids")
        asks = obj.get("asks")
        if bid is None and isinstance(bids, list) and bids:
            top = bids[0]
            bid = parse_decimal(top.get("price") if isinstance(top, dict) else top[0] if isinstance(top, list) and top else None)
        if ask is None and isinstance(asks, list) and asks:
            top = asks[0]
            ask = parse_decimal(top.get("price") if isinstance(top, dict) else top[0] if isinstance(top, list) and top else None)

        if asset_id is not None and (bid is not None or ask is not None or mid is not None):
            updates.append((str(asset_id), bid, ask, mid))

        for key in ("data", "event", "events", "book", "books", "payload"):
            nested = obj.get(key)
            if isinstance(nested, list):
                for item in nested:
                    process_obj(item)
            else:
                process_obj(nested)

    process_obj(payload)
    return updates


def seed_midpoint_snapshot(settings: Settings, tracker: BookTracker, asset_ids: list[str]) -> None:
    if not settings.use_midpoint_snapshot:
        return
    for asset_id in asset_ids:
        try:
            url = f"{settings.clob_host.rstrip('/')}/midpoint"
            r = requests.get(url, params={"asset_id": asset_id}, timeout=settings.request_timeout_seconds)
            if r.status_code >= 400:
                continue
            payload = r.json()
            mid = parse_decimal(payload.get("mid") if isinstance(payload, dict) else None)
            if mid is not None:
                tracker.upsert(asset_id, None, None, mid)
        except Exception:
            continue


class CLOBWebSocketClient:
    def __init__(self, settings: Settings, asset_ids: list[str], tracker: BookTracker) -> None:
        self.settings = settings
        self.asset_ids = asset_ids
        self.tracker = tracker
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ws = websocket.WebSocket()
                ws.connect(self.settings.clob_ws_url, timeout=self.settings.request_timeout_seconds)
                ws.settimeout(self.settings.ws_recv_timeout_seconds)
                sub = {"type": "subscribe", "channel": "market", "asset_ids": self.asset_ids, "assets_ids": self.asset_ids}
                ws.send(json.dumps(sub))
                logging.info("Subscribed to CLOB ws for %d asset_ids", len(self.asset_ids))

                last_heartbeat = time.time()
                while not self._stop.is_set():
                    try:
                        raw = ws.recv()
                        payload = json.loads(raw)
                        for aid, bid, ask, mid in parse_book_message(payload):
                            self.tracker.upsert(aid, bid, ask, mid)
                    except WebSocketTimeoutException:
                        now = time.time()
                        if now - last_heartbeat >= self.settings.ws_heartbeat_seconds:
                            try:
                                ws.ping("keepalive")
                                last_heartbeat = now
                                logging.debug("WebSocket heartbeat ping sent")
                                continue
                            except Exception:
                                logging.warning("WebSocket ping failed; reconnecting")
                                break
            except Exception:
                logging.exception("WebSocket disconnected; reconnecting in %ss", self.settings.ws_reconnect_seconds)
                time.sleep(self.settings.ws_reconnect_seconds)


class VolatilityEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[str, tuple[float, Decimal]] = {}

    def _binance_symbol(self, underlying: str) -> str | None:
        return {
            "BTC": "BTCUSDT",
            "BITCOIN": "BTCUSDT",
            "ETH": "ETHUSDT",
            "ETHEREUM": "ETHUSDT",
            "SOL": "SOLUSDT",
            "SOLANA": "SOLUSDT",
        }.get(underlying.upper())

    def _coingecko_coin(self, underlying: str) -> str | None:
        return {
            "BTC": "bitcoin",
            "BITCOIN": "bitcoin",
            "ETH": "ethereum",
            "ETHEREUM": "ethereum",
            "SOL": "solana",
            "SOLANA": "solana",
        }.get(underlying.upper())

    def fetch_spot_price_usd(self, underlying: str) -> Decimal | None:
        symbol = self._binance_symbol(underlying)
        if not symbol:
            return None
        try:
            r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=self.settings.request_timeout_seconds)
            r.raise_for_status()
            return parse_decimal(r.json().get("price"))
        except Exception:
            logging.exception("Failed spot fetch for %s", underlying)
            return None

    def _rolling_daily_vol(self, underlying: str) -> Decimal | None:
        now = time.time()
        ck = f"{self.settings.volatility_source}:{underlying.upper()}"
        cached = self._cache.get(ck)
        if cached and now - cached[0] < self.settings.volatility_update_seconds:
            return cached[1]

        points: list[tuple[float, float]] = []
        try:
            if self.settings.volatility_source == "binance":
                symbol = self._binance_symbol(underlying)
                if not symbol:
                    return None
                limit = max(20, min(1000, self.settings.volatility_rolling_window_points + 5))
                r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": symbol, "interval": "1m", "limit": limit}, timeout=self.settings.request_timeout_seconds)
                r.raise_for_status()
                rows = r.json()
                for row in rows:
                    if isinstance(row, list) and len(row) >= 5:
                        ts = float(row[0]) / 1000.0
                        px = float(row[4])
                        if px > 0:
                            points.append((ts, px))
            else:
                coin = self._coingecko_coin(underlying)
                if not coin:
                    return None
                r = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart", params={"vs_currency": "usd", "days": "1", "interval": ""}, timeout=self.settings.request_timeout_seconds)
                r.raise_for_status()
                prices = r.json().get("prices", [])
                for row in prices:
                    if isinstance(row, list) and len(row) >= 2:
                        ts = float(row[0]) / 1000.0
                        px = float(row[1])
                        if px > 0:
                            points.append((ts, px))

            if len(points) < 5:
                return None
            window = points[-self.settings.volatility_rolling_window_points :]
            returns: list[float] = []
            dts: list[float] = []
            for i in range(1, len(window)):
                t0, p0 = window[i - 1]
                t1, p1 = window[i]
                if p0 <= 0 or p1 <= 0 or t1 <= t0:
                    continue
                returns.append(math.log(p1 / p0))
                dts.append(t1 - t0)
            if len(returns) < 4:
                return None
            mean = sum(returns) / len(returns)
            variance = sum((x - mean) ** 2 for x in returns) / len(returns)
            avg_dt = sum(dts) / len(dts)
            steps_per_day = 86400.0 / avg_dt if avg_dt > 0 else 0
            if steps_per_day <= 0:
                return None
            daily_vol = Decimal(str(math.sqrt(variance * steps_per_day)))
            self._cache[ck] = (now, daily_vol)
            return daily_vol
        except Exception:
            logging.exception("Failed rolling volatility fetch for %s", underlying)
            return None

    def duration_adjusted_vol(self, underlying: str, end_ts: datetime | None) -> Decimal | None:
        if not self.settings.enable_volatility_context:
            return None
        daily = self._rolling_daily_vol(underlying)
        if daily is None:
            return None
        if end_ts is None:
            return daily
        hours = max(0.5, (end_ts - datetime.now(timezone.utc)).total_seconds() / 3600.0)
        t_days = Decimal(str(hours / 24.0))
        return daily * Decimal(str(math.sqrt(float(t_days))))


def group_event_key(m: ThresholdMarket) -> str:
    return f"{m.underlying}|{m.direction}|{m.event_name}"


def evaluate_signals(settings: Settings, tracker: BookTracker, markets: list[ThresholdMarket], vol_engine: VolatilityEngine) -> list[PinchSignal]:
    grouped: dict[str, list[ThresholdMarket]] = {}
    for m in markets:
        grouped.setdefault(group_event_key(m), []).append(m)

    signals: list[PinchSignal] = []
    for key, rows in grouped.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: r.level)
        spot = vol_engine.fetch_spot_price_usd(rows[0].underlying)
        if not spot or spot <= 0:
            continue

        probs: list[tuple[ThresholdMarket, Decimal]] = []
        for row in rows:
            p = tracker.get_mid(row.yes_asset_id)
            if p is None:
                continue
            p = max(Decimal("0"), min(Decimal("1"), p))
            probs.append((row, p))

        for i in range(len(probs) - 1):
            low, p_low = probs[i]
            high, p_high = probs[i + 1]
            band_prob = p_low - p_high
            band_width_pct = (high.level - low.level) / spot
            pinch = band_prob / max(band_width_pct, Decimal("0.000001"))

            monotonic_violation = (p_high - p_low) > settings.monotonicity_tolerance
            narrow_overweight = band_prob > settings.narrow_band_prob_threshold and band_width_pct < settings.narrow_band_width_pct_threshold
            high_pinch = pinch > settings.pinch_score_threshold
            if not (monotonic_violation or narrow_overweight or high_pinch):
                continue

            if monotonic_violation:
                reason = "monotonicity_violation"
                suggestion = "BUY lower-strike YES / SELL higher-strike YES (restore monotonic CDF)."
            elif band_prob > 0:
                reason = "pinched_band"
                suggestion = "SELL lower-strike YES / BUY higher-strike YES (curve decompression)."
            else:
                reason = "negative_band"
                suggestion = "BUY lower-strike YES / SELL higher-strike YES (inverted cheap band)."

            hours_to_event = None
            if low.end_ts is not None:
                hours_to_event = max(0.0, (low.end_ts - datetime.now(timezone.utc)).total_seconds() / 3600.0)
            dur_vol = vol_engine.duration_adjusted_vol(low.underlying, low.end_ts)

            signals.append(PinchSignal(
                event_key=key,
                event_name=low.event_name,
                underlying=low.underlying,
                level_low=low.level,
                level_high=high.level,
                low_asset_id=low.yes_asset_id,
                high_asset_id=high.yes_asset_id,
                p_low=p_low,
                p_high=p_high,
                band_prob=band_prob,
                band_width_pct=band_width_pct,
                pinch=pinch,
                reason=reason,
                suggestion=suggestion,
                hours_to_event=hours_to_event,
                duration_scaled_vol=dur_vol,
            ))
    return signals


def log_near_miss_diagnostics(settings: Settings, tracker: BookTracker, markets: list[ThresholdMarket], vol_engine: VolatilityEngine) -> None:
    if not settings.enable_diagnostics:
        return

    candidates: list[tuple[Decimal, str]] = []
    grouped: dict[str, list[ThresholdMarket]] = {}
    for m in markets:
        grouped.setdefault(group_event_key(m), []).append(m)

    for _, rows in grouped.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: r.level)
        spot = vol_engine.fetch_spot_price_usd(rows[0].underlying)
        if not spot or spot <= 0:
            continue

        probs: list[tuple[ThresholdMarket, Decimal]] = []
        for row in rows:
            p = tracker.get_mid(row.yes_asset_id)
            if p is None:
                continue
            p = max(Decimal("0"), min(Decimal("1"), p))
            probs.append((row, p))

        for i in range(len(probs) - 1):
            low, p_low = probs[i]
            high, p_high = probs[i + 1]
            band_prob = p_low - p_high
            band_width_pct = (high.level - low.level) / spot
            pinch = band_prob / max(band_width_pct, Decimal("0.000001"))

            mono_gap = settings.monotonicity_tolerance - (p_high - p_low)
            band_gap = settings.narrow_band_prob_threshold - band_prob
            pinch_gap = settings.pinch_score_threshold - pinch
            score = min(mono_gap, band_gap, pinch_gap)

            line = (
                "Diag near-miss | event=%s | %s->%s | p_low=%.4f p_high=%.4f | "
                "band_prob=%s (gap=%s) | width=%s | pinch=%s (gap=%s) | mono_gap=%s"
            ) % (
                low.event_name,
                low.level,
                high.level,
                float(p_low),
                float(p_high),
                band_prob,
                band_gap,
                band_width_pct,
                pinch,
                pinch_gap,
                mono_gap,
            )
            candidates.append((score, line))

    if not candidates:
        logging.info("Diagnostics: no eligible adjacent threshold pairs this cycle")
        return

    candidates.sort(key=lambda x: x[0])
    top = candidates[: settings.diagnostics_top_n]
    logging.info("Diagnostics: top %d near-miss pairs", len(top))
    for _, line in top:
        logging.info(line)



class ExecutionEngine:
    def __init__(self, settings: Settings, store: SpreadStore) -> None:
        self.settings = settings
        self.store = store
        self.session_realized_pnl = Decimal("0")
        self.client: ClobClient | None = None
        if settings.enable_trading:
            self.client = self._init_client()

    def _init_client(self) -> ClobClient:
        client = ClobClient(
            host=self.settings.clob_host,
            key=self.settings.private_key,
            chain_id=self.settings.chain_id,
            signature_type=2,
            funder=self.settings.funder_address,
        )
        if self.settings.api_key and self.settings.api_secret and self.settings.api_passphrase:
            client.set_api_creds(ApiCreds(api_key=self.settings.api_key, api_secret=self.settings.api_secret, api_passphrase=self.settings.api_passphrase))
        else:
            client.set_api_creds(client.create_or_derive_api_creds())
        logging.info("Execution enabled on CLOB host %s", self.settings.clob_host)
        return client

    def _order(self, token_id: str, side: str, price: Decimal, qty: Decimal) -> None:
        if self.client is None:
            return
        order = OrderArgs(token_id=token_id, price=float(price), size=float(qty), side=side)
        signed = self.client.create_order(order)
        self.client.post_order(signed, OrderType.GTC)

    def _qty(self, price: Decimal) -> Decimal:
        if price <= 0:
            return Decimal("0")
        return (self.settings.trade_usd_per_leg / price).quantize(Decimal("0.0001"))

    def _signed_pnl(self, side: str, entry: Decimal, mark: Decimal, qty: Decimal) -> Decimal:
        if side == "BUY":
            return (mark - entry) * qty
        return (entry - mark) * qty

    def pnl_snapshot(self, tracker: BookTracker) -> tuple[Decimal, Decimal, Decimal]:
        unrealized = Decimal("0")
        for pos in self.store.positions.values():
            low_mid = tracker.get_mid(pos.low_asset_id)
            high_mid = tracker.get_mid(pos.high_asset_id)
            if low_mid is not None:
                unrealized += self._signed_pnl(pos.low_side, pos.low_entry_price, low_mid, pos.low_qty)
            if high_mid is not None:
                unrealized += self._signed_pnl(pos.high_side, pos.high_entry_price, high_mid, pos.high_qty)
        total = self.session_realized_pnl + unrealized
        return self.session_realized_pnl, unrealized, total

    def log_pnl_summary(self, tracker: BookTracker) -> None:
        realized, unrealized, total = self.pnl_snapshot(tracker)
        mode = "live" if self.settings.enable_trading else "paper"
        logging.info("PnL | mode=%s | realized=%s | unrealized=%s | total=%s | open_spreads=%d", mode, realized, unrealized, total, len(self.store.positions))

    def maybe_enter(self, sig: PinchSignal) -> None:
        key = f"{sig.event_key}|{sig.level_low}|{sig.level_high}"
        if key in self.store.positions:
            return
        if len(self.store.positions) >= self.settings.max_open_spreads:
            return

        if sig.reason in {"pinched_band", "monotonicity_violation"}:
            low_side, high_side = "SELL", "BUY"
            low_price = max(Decimal("0.01"), sig.p_low - self.settings.price_buffer)
            high_price = min(Decimal("0.99"), sig.p_high + self.settings.price_buffer)
        else:
            low_side, high_side = "BUY", "SELL"
            low_price = min(Decimal("0.99"), sig.p_low + self.settings.price_buffer)
            high_price = max(Decimal("0.01"), sig.p_high - self.settings.price_buffer)

        low_qty = self._qty(low_price)
        high_qty = self._qty(high_price)
        if low_qty < self.settings.min_order_size or high_qty < self.settings.min_order_size:
            return

        try:
            if self.settings.enable_trading:
                self._order(sig.low_asset_id, low_side, low_price, low_qty)
                self._order(sig.high_asset_id, high_side, high_price, high_qty)

            if self.settings.enable_trading or self.settings.enable_paper_execution:
                self.store.upsert(
                    SpreadPosition(
                        key=key,
                        event_key=sig.event_key,
                        low_asset_id=sig.low_asset_id,
                        high_asset_id=sig.high_asset_id,
                        low_side=low_side,
                        high_side=high_side,
                        low_qty=low_qty,
                        high_qty=high_qty,
                        low_entry_price=low_price,
                        high_entry_price=high_price,
                        entry_band_prob=sig.band_prob,
                        entry_pinch=sig.pinch,
                        opened_at=time.time(),
                    )
                )
                mode = "live" if self.settings.enable_trading else "paper"
                logging.warning("🚨 TRADE ENTERED | spread=%s | mode=%s | low=%s@%s (%s) | high=%s@%s (%s)", key, mode, low_qty, low_price, low_side, high_qty, high_price, high_side)
        except Exception:
            logging.exception("Failed entering spread %s", key)

    def maybe_exit(self, sig: PinchSignal) -> None:
        key = f"{sig.event_key}|{sig.level_low}|{sig.level_high}"
        pos = self.store.positions.get(key)
        if not pos:
            return

        drop_target = pos.entry_band_prob * (Decimal("1") - self.settings.correction_drop_pct)
        should_exit = sig.band_prob <= drop_target or sig.pinch <= self.settings.correction_pinch_below
        if not should_exit:
            return

        try:
            if pos.low_side == "BUY":
                low_exit_side = "SELL"
                low_exit_price = max(Decimal("0.01"), sig.p_low - self.settings.price_buffer)
            else:
                low_exit_side = "BUY"
                low_exit_price = min(Decimal("0.99"), sig.p_low + self.settings.price_buffer)

            if pos.high_side == "BUY":
                high_exit_side = "SELL"
                high_exit_price = max(Decimal("0.01"), sig.p_high - self.settings.price_buffer)
            else:
                high_exit_side = "BUY"
                high_exit_price = min(Decimal("0.99"), sig.p_high + self.settings.price_buffer)

            if self.settings.enable_trading:
                self._order(pos.low_asset_id, low_exit_side, low_exit_price, pos.low_qty)
                self._order(pos.high_asset_id, high_exit_side, high_exit_price, pos.high_qty)

            realized_delta = self._signed_pnl(pos.low_side, pos.low_entry_price, low_exit_price, pos.low_qty) + self._signed_pnl(pos.high_side, pos.high_entry_price, high_exit_price, pos.high_qty)
            self.session_realized_pnl += realized_delta

            self.store.remove(key)
            mode = "live" if self.settings.enable_trading else "paper"
            logging.warning("✅ TRADE EXITED | spread=%s | mode=%s | realized_delta=%s | band_prob=%s | pinch=%s", key, mode, realized_delta, sig.band_prob, sig.pinch)
        except Exception:
            logging.exception("Failed exiting spread %s", key)


def format_signal(sig: PinchSignal, settings: Settings) -> str:
    drop_target = (sig.band_prob * (Decimal("1") - settings.correction_drop_pct)).quantize(Decimal("0.0001"))
    vol_line = f"Duration-scaled vol: {sig.duration_scaled_vol}\n" if sig.duration_scaled_vol is not None else "Duration-scaled vol: n/a\n"
    time_line = f"Hours to event end: {sig.hours_to_event:.2f}\n" if sig.hours_to_event is not None else "Hours to event end: n/a\n"
    return (
        f"ALERT [{sig.reason}] {sig.event_name}\n"
        f"Underlying: {sig.underlying}\n"
        f"Thresholds: {sig.level_low} -> {sig.level_high}\n"
        f"Probabilities: p_low={sig.p_low:.4f}, p_high={sig.p_high:.4f}\n"
        f"Band prob: {sig.band_prob:.6f}\n"
        f"Band width pct of spot: {sig.band_width_pct:.6f}\n"
        f"Pinch score: {sig.pinch:.4f}\n"
        f"{time_line}"
        f"{vol_line}"
        f"Trade idea: {sig.suggestion}\n"
        "Exit guidance (curve correction, direction-agnostic): "
        f"close when band_prob <= {drop_target} OR pinch <= {settings.correction_pinch_below}."
    )


def notify_discord(settings: Settings, message: str) -> None:
    if not settings.enable_discord_alerts or not settings.discord_webhook_url:
        return
    r = requests.post(settings.discord_webhook_url, json={"content": message[:1900]}, timeout=settings.request_timeout_seconds)
    r.raise_for_status()


def run() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)

    should_stop = False

    def _handle(signum: int, _frame: Any) -> None:
        nonlocal should_stop
        should_stop = True
        logging.info("Received signal %s, shutting down...", signum)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    markets = discover_threshold_markets(settings)
    asset_ids = sorted({m.yes_asset_id for m in markets})

    tracker = BookTracker()
    seed_midpoint_snapshot(settings, tracker, asset_ids)

    ws_client = CLOBWebSocketClient(settings, asset_ids, tracker)
    ws_client.start()

    vol_engine = VolatilityEngine(settings)
    spread_store = SpreadStore(settings.positions_file)
    exec_engine = ExecutionEngine(settings, spread_store)

    last_alert_at: dict[str, float] = {}
    last_refresh_at = time.time()
    last_console_clear = 0.0
    last_dashboard_at = 0.0
    cycle = 0

    while not should_stop:
        try:
            cycle += 1
            if settings.clear_console_seconds > 0 and sys.stdout.isatty():
                now_clear = time.time()
                if now_clear - last_console_clear >= settings.clear_console_seconds:
                    os.system("cls" if os.name == "nt" else "clear")
                    last_console_clear = now_clear
            now = time.time()
            if now - last_refresh_at >= settings.poll_interval_seconds:
                markets = discover_threshold_markets(settings)
                asset_ids = sorted({m.yes_asset_id for m in markets})
                seed_midpoint_snapshot(settings, tracker, asset_ids)
                last_refresh_at = now

            signals = evaluate_signals(settings, tracker, markets, vol_engine)
            if not signals:
                logging.info("No pinch / CDF inconsistency alerts in this cycle")
                log_near_miss_diagnostics(settings, tracker, markets, vol_engine)

            realized, unrealized, total = exec_engine.pnl_snapshot(tracker)
            exec_engine.log_pnl_summary(tracker)
            if now - last_dashboard_at >= settings.dashboard_interval_seconds:
                log_cycle_dashboard(
                    settings=settings,
                    cycle=cycle,
                    markets_count=len(markets),
                    asset_count=len(asset_ids),
                    signal_count=len(signals),
                    open_spreads=len(spread_store.positions),
                    realized=realized,
                    unrealized=unrealized,
                    total=total,
                )
                last_dashboard_at = now

            for sig in signals:
                key = f"{sig.event_key}|{sig.level_low}|{sig.level_high}|{sig.reason}"
                msg = format_signal(sig, settings)
                exec_engine.maybe_enter(sig)
                exec_engine.maybe_exit(sig)

                if now - last_alert_at.get(key, 0) >= settings.alert_cooldown_seconds:
                    logging.warning(msg)
                    try:
                        notify_discord(settings, msg)
                    except Exception:
                        logging.exception("Failed sending Discord alert")
                    last_alert_at[key] = now

        except Exception:
            logging.exception("Runtime error in monitor loop")

        time.sleep(settings.eval_interval_seconds)

    ws_client.stop()


def main() -> int:
    try:
        run()
        return 0
    except ConfigError as exc:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(message)s")
        logging.error("Configuration error: %s", exc)
        return 2
    except Exception:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(message)s")
        logging.exception("Fatal startup error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
