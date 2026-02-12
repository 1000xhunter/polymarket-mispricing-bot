# Polymarket Mispricing Bot (Python 3.11)

A Python 3.11 bot that scans active Polymarket markets, detects mispricings (`YES + NO != 1`), and can optionally auto-execute strategy trades on the Polymarket CLOB API with wallet authentication, position tracking, exits, and risk controls.

## Features

- Continuous scanning of active Polymarket markets (paged API polling).
- Mispricing detection with configurable thresholds.
- Optional Discord alerts.
- Optional trade execution via Polymarket CLOB API.
- Position tracking persisted to `positions.json`.
- Auto-entry and auto-exit rules (normalization, take-profit, and stop-loss).
- Optional dynamic sizing layer for per-leg notional.
- Risk management:
  - max active positions
  - max total exposure
  - max per-market exposure
  - max unrealized loss gate
- Robust logging and exception handling.
- Docker + Compose for 24/7 deployment.

## Folder structure

```text
.
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── docker-compose.yml
├── main.py
└── requirements.txt
```

## Strategy behavior

### Detection
- Bot fetches active markets each cycle and computes mispricing by:
  - `total = YES_price + NO_price`
  - `deviation = abs(1 - total)`
- It flags markets where `deviation >= MISPRICING_THRESHOLD`.

### Entry execution (BUY trigger)
A BUY sequence is triggered only when **all** conditions are true:
- `ENABLE_TRADING=true`.
- Mispricing exists and is on the underpriced side: `total < (1 - MISPRICING_THRESHOLD)`.
- Market has valid YES/NO CLOB token IDs.
- No existing tracked YES/NO position already exists for that market.
- Risk checks pass (`MAX_ACTIVE_POSITIONS`, `MAX_TOTAL_EXPOSURE_USD`, `MAX_MARKET_EXPOSURE_USD`, and unrealized-loss gate `MAX_LOSS_USD`).
- Calculated order size for each leg is >= `MIN_ORDER_SIZE`.

If true, bot submits **BUY YES** and **BUY NO** orders using a buffered price (`market_price + PRICE_BUFFER`, capped at `0.99`). Per-leg notional is derived from `TRADE_USD_PER_LEG`, then optionally adjusted by dynamic sizing and volatility sizing before converting to quantity (`usd_per_leg / price`).

### Exit execution (SELL trigger)
A SELL sequence is triggered when:
- `ENABLE_TRADING=true`.
- There is at least one tracked position leg (YES/NO) for that market.
- **Any one** of the exit conditions is hit:
  - Market normalization: `abs(1 - total) <= EXIT_THRESHOLD`.
  - Per-market take profit: unrealized PnL for that market `>= TAKE_PROFIT_USD`.
  - Per-market stop loss: unrealized PnL for that market `<= -STOP_LOSS_USD`.

If true, bot submits **SELL** orders for tracked quantities using buffered sell prices (`market_price - PRICE_BUFFER`, floored at `0.01`), then removes those legs from local position tracking.

### Dynamic sizing (optional)
When `ENABLE_DYNAMIC_SIZING=true`, the bot adjusts USD size-per-leg before placing orders:
- Starts with base: `TRADE_USD_PER_LEG * volatility_multiplier`.
- Scales by signal strength (`deviation / MISPRICING_THRESHOLD`, clamped).
- Scales down as portfolio exposure approaches `MAX_TOTAL_EXPOSURE_USD`.
- Applies cap by exposure percent using `DYNAMIC_SIZE_EXPOSURE_PCT` (easy to tweak later for your preferred max position % behavior).
- Clamps final size between `DYNAMIC_SIZE_MIN_USD_PER_LEG` and `DYNAMIC_SIZE_MAX_USD_PER_LEG`.

### Real-time scanning + market identification
- The loop polls active/open markets continuously and tracks newly seen market IDs each cycle.
- Underlying asset is inferred from market text (currently BTC/ETH/SOL mappings for volatility sizing).
- If market metadata includes an event end timestamp (for example `endDate`/`closeTime`), the bot computes time-to-event and adjusts the volatility sizing scale for shorter/longer horizons.

### Volatility handling (BTC/ETH/etc.)
- Yes, now it can account for underlying volatility **for sizing** when `ENABLE_VOLATILITY_SIZING=true`.
- The bot infers underlying (currently BTC/ETH/SOL) from market question text.
- Default source is CoinGecko (`VOLATILITY_SOURCE=coingecko`), and Binance is supported (`VOLATILITY_SOURCE=binance`) using 1-minute klines.
- It fetches **rolling intraday prices** from a configurable source and recomputes realized volatility on a rolling basis; then scales `TRADE_USD_PER_LEG` by: `target_vol / realized_vol`.
- Multiplier is clamped by `MIN_VOLATILITY_MULTIPLIER` and `MAX_VOLATILITY_MULTIPLIER` for safety.
- It also applies a lightweight timeframe scaler inferred from market wording (e.g., intraday/weekly/monthly/yearly) so volatility sizing better matches the event horizon.
- If volatility lookup fails or underlying is unknown, it safely falls back to base size (multiplier = 1).

### Risk controls
- `MAX_ACTIVE_POSITIONS`
- `MAX_TOTAL_EXPOSURE_USD`
- `MAX_MARKET_EXPOSURE_USD`
- `MAX_LOSS_USD` (blocks new entries when unrealized PnL breaches this loss level)

## Environment variables

Copy `.env.example` to `.env` and set values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GAMMA_API_URL` | No | `https://gamma-api.polymarket.com/markets` | Gamma markets endpoint |
| `POLL_INTERVAL_SECONDS` | No | `20` | Scan loop interval |
| `MISPRICING_THRESHOLD` | No | `0.03` | Entry detection threshold |
| `EXIT_THRESHOLD` | No | `0.01` | Exit normalization threshold |
| `REQUEST_TIMEOUT_SECONDS` | No | `15` | HTTP timeout |
| `MAX_MARKETS_PER_SCAN` | No | `1000` | Max active markets scanned per cycle |
| `PAGE_SIZE` | No | `200` | Page size for Gamma API |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `DISCORD_WEBHOOK_URL` | No | empty | Optional webhook alerts |
| `ENABLE_TRADING` | No | `false` | Enable/disable order execution |
| `POSITIONS_FILE` | No | `positions.json` | Local persisted position state |
| `CLOB_HOST` | Yes (if trading) | `https://clob.polymarket.com` | CLOB API host |
| `CHAIN_ID` | Yes (if trading) | `137` | Network chain id |
| `POLYMARKET_PRIVATE_KEY` | Yes (if trading) | empty | Wallet private key |
| `POLYMARKET_FUNDER_ADDRESS` | Optional | empty | Funder/proxy address when applicable |
| `POLYMARKET_API_KEY` | Optional | empty | Existing CLOB API key |
| `POLYMARKET_API_SECRET` | Optional | empty | Existing CLOB API secret |
| `POLYMARKET_API_PASSPHRASE` | Optional | empty | Existing CLOB API passphrase |
| `MAX_ACTIVE_POSITIONS` | No | `20` | Max tracked position legs |
| `MAX_TOTAL_EXPOSURE_USD` | No | `500` | Total exposure cap |
| `MAX_MARKET_EXPOSURE_USD` | No | `100` | Per-market exposure cap |
| `MAX_LOSS_USD` | No | `100` | Max unrealized loss gate for blocking new entries |
| `TAKE_PROFIT_USD` | No | `15` | Per-market unrealized PnL take-profit trigger |
| `STOP_LOSS_USD` | No | `15` | Per-market unrealized PnL stop-loss trigger |
| `TRADE_USD_PER_LEG` | No | `10` | Base USD notional per order leg |
| `ENABLE_DYNAMIC_SIZING` | No | `false` | Enable dynamic per-leg sizing |
| `DYNAMIC_SIZE_EXPOSURE_PCT` | No | `10` | Max per-trade budget as % of `MAX_TOTAL_EXPOSURE_USD` |
| `DYNAMIC_SIZE_MIN_USD_PER_LEG` | No | `5` | Lower clamp for dynamic USD-per-leg |
| `DYNAMIC_SIZE_MAX_USD_PER_LEG` | No | `50` | Upper clamp for dynamic USD-per-leg |
| `MIN_ORDER_SIZE` | No | `1` | Minimum order size |
| `PRICE_BUFFER` | No | `0.01` | Price adjustment on entry/exit |
| `ENABLE_VOLATILITY_SIZING` | No | `false` | Enable volatility-adjusted position sizing |
| `VOLATILITY_LOOKBACK_DAYS` | No | `14` | Legacy fallback lookback setting (kept for compatibility) |
| `TARGET_DAILY_VOLATILITY` | No | `0.03` | Target daily volatility used in sizing ratio |
| `MIN_VOLATILITY_MULTIPLIER` | No | `0.25` | Lower clamp for volatility multiplier |
| `MAX_VOLATILITY_MULTIPLIER` | No | `1.50` | Upper clamp for volatility multiplier |
| `VOLATILITY_UPDATE_SECONDS` | No | `60` | Rolling volatility refresh interval per underlying asset |
| `VOLATILITY_ROLLING_WINDOW_POINTS` | No | `120` | Number of latest intraday price points used for rolling vol |

---

## Super simple local setup (Windows, no GitHub needed)

If you already have this project folder on your computer, do this:

1) Open the project folder in File Explorer.
2) Click the address bar, type `powershell`, press Enter.
3) Copy/paste these commands **one by one**:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

4) In Notepad (`.env` file):
- Leave `ENABLE_TRADING=false` for your first run (safe mode, no real trades).
- Save and close Notepad.

5) Start the bot:

```powershell
python main.py
```

6) Stop the bot anytime with `Ctrl + C`.

### What you should see
- Lines like: scanned markets, mispricing checks, and loop timing.
- If you set a Discord webhook, alert messages when mispricings are found.

### If something fails
- If `py -3.11` fails, install Python 3.11 from python.org and check “Add Python to PATH” during install.
- If script activation is blocked, run this once and retry:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

- If package install fails, retry:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Run with Docker

1. Create env file:

```powershell
Copy-Item .env.example .env
```

2. Build and run:

```powershell
docker compose up -d --build
```

3. Tail logs:

```powershell
docker compose logs -f
```

4. Stop:

```powershell
docker compose down
```

## Safety note

Automated trading is risky. Test first with `ENABLE_TRADING=false`, validate signals and logs, then enable trading only with conservative risk limits.
