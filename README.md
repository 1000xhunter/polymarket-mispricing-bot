# Polymarket Probability Pinch Monitor + Execution (Python 3.11)

A real-time, non-directional Polymarket tool that detects adjacent-threshold CDF inconsistencies (probability pinching / monotonicity breaks), adds **time-aware volatility context**, and can optionally execute relative-value spread trades.

## Folder structure

```text
.
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── docker-compose.yml
├── main.py
├── requirements.txt
└── run_local.ps1
```

## What it does

For events like `BTC above 69500`, `BTC above 69750`, `BTC above 70000`, the bot:

1. Discovers markets from Gamma API and extracts YES token IDs.
2. Subscribes to CLOB websocket (`/ws/market`) and maintains live YES bid/ask/mid.
3. Converts YES midpoint into implied probability `p(level)`.
4. Computes for adjacent levels:
   - `band_prob = p(low) - p(high)`
   - `band_width_pct = (level_high - level_low) / spot`
   - `pinch = band_prob / max(band_width_pct, 1e-6)`
5. Flags signals if:
   - narrow band carries too much probability mass,
   - pinch score exceeds threshold,
   - monotonicity is violated (`p(high) > p(low)` beyond tolerance).
6. Computes **time to event end** (when end timestamp exists) and **duration-scaled volatility** from rolling crypto data.
7. Emits alerts with trade suggestion + exit guidance.
8. Optionally executes spread trades (paper or live):
   - enters when signal appears,
   - exits when correction criteria are met (`band_prob` drop or pinch below threshold).
9. When no alert triggers, diagnostics mode prints the closest near-miss pairs so you can tune thresholds.
10. Prints a per-cycle P&L tracker (realized/unrealized/total) for open spread positions.
11. Makes trade actions visually obvious in logs (`🚨 TRADE ENTERED`, `✅ TRADE EXITED`) and can auto-clear console on interval.

## Environment variables

Copy `.env.example` to `.env` and edit.

| Variable | Default | Description |
|---|---:|---|
| `GAMMA_API_URL` | `https://gamma-api.polymarket.com/markets` | Gamma discovery endpoint |
| `CLOB_HOST` | `https://clob.polymarket.com` | CLOB REST host |
| `CLOB_WS_URL` | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | CLOB market websocket |
| `WS_RECV_TIMEOUT_SECONDS` | `20` | WebSocket read timeout before heartbeat/ping logic |
| `WS_HEARTBEAT_SECONDS` | `15` | Interval for keepalive ping when idle |
| `WS_RECONNECT_SECONDS` | `3` | Delay before websocket reconnect attempts |
| `CLEAR_CONSOLE_SECONDS` | `60` | Auto-clear console interval in seconds (`0` disables) |
| `DASHBOARD_INTERVAL_SECONDS` | `10` | How often to print the compact runtime dashboard |
| `ENABLE_WINDOWS_TITLE` | `true` | Update PowerShell/CMD window title with mode/signals/P&L |
| `REQUEST_TIMEOUT_SECONDS` | `15` | HTTP/WebSocket timeout |
| `POLL_INTERVAL_SECONDS` | `60` | Market rediscovery interval |
| `EVAL_INTERVAL_SECONDS` | `2` | Signal evaluation interval |
| `MAX_MARKETS_PER_SCAN` | `2000` | Max markets each scan |
| `PAGE_SIZE` | `200` | Gamma page size |
| `CRYPTO_ONLY` | `true` | Restrict to crypto-like markets |
| `CRYPTO_KEYWORDS` | `btc,bitcoin,eth,ethereum,sol,solana,crypto,doge,xrp` | Crypto filter keywords |
| `USE_MIDPOINT_SNAPSHOT` | `true` | Seed mids via CLOB `/midpoint` |
| `PINCH_SCORE_THRESHOLD` | `0.8` | Pinch trigger threshold |
| `NARROW_BAND_PROB_THRESHOLD` | `0.35` | High band prob trigger |
| `NARROW_BAND_WIDTH_PCT_THRESHOLD` | `0.005` | Narrow width trigger (<0.5%) |
| `MONOTONICITY_TOLERANCE` | `0.01` | Allowed monotonic slack |
| `CORRECTION_DROP_PCT` | `0.30` | Exit when band_prob drops by this % |
| `CORRECTION_PINCH_BELOW` | `0.4` | Exit when pinch drops below this |
| `ENABLE_VOLATILITY_CONTEXT` | `true` | Compute rolling volatility context |
| `VOLATILITY_SOURCE` | `binance` | `binance` or `coingecko` |
| `VOLATILITY_UPDATE_SECONDS` | `60` | Vol cache refresh interval |
| `VOLATILITY_ROLLING_WINDOW_POINTS` | `120` | Points used in rolling vol calc |
| `ENABLE_DISCORD_ALERTS` | `false` | Enable Discord notifications |
| `DISCORD_WEBHOOK_URL` | empty | Discord webhook URL |
| `ENABLE_TRADING` | `false` | Live execution toggle |
| `ENABLE_PAPER_EXECUTION` | `true` | Paper spread tracking when not live |
| `TRADE_USD_PER_LEG` | `10` | Notional per spread leg |
| `MIN_ORDER_SIZE` | `1` | Minimum order quantity |
| `PRICE_BUFFER` | `0.01` | Price buffer on entries/exits |
| `MAX_OPEN_SPREADS` | `30` | Max concurrent spread positions |
| `POSITIONS_FILE` | `spread_positions.json` | Local spread state file |
| `CHAIN_ID` | `137` | Chain id for CLOB client |
| `POLYMARKET_PRIVATE_KEY` | empty | Required when `ENABLE_TRADING=true` |

---

## Quick one-command run (Windows)

```powershell
powershell -NoExit -ExecutionPolicy Bypass -File .\run_local.ps1
```

Faster rerun:

```powershell
powershell -NoExit -ExecutionPolicy Bypass -File .\run_local.ps1 -SkipInstall
```

Setup only:

```powershell
powershell -NoExit -ExecutionPolicy Bypass -File .\run_local.ps1 -SetupOnly
```


If you see `param : The term 'param' is not recognized`:
- update to latest `run_local.ps1` (it must start with `param(...)` on line 1),
- then run from an open PowerShell in the project folder:

```powershell
.\run_local.ps1
```

or:

```powershell
powershell -NoExit -ExecutionPolicy Bypass -File .\run_local.ps1
```

## Manual Windows run

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python main.py
```

## Docker run

```powershell
Copy-Item .env.example .env
docker compose up -d --build
docker compose logs -f
```

Stop:

```powershell
docker compose down
```

## Safety

Start with:
- `ENABLE_TRADING=false`
- `ENABLE_PAPER_EXECUTION=true`

Validate signal quality, exits, and logs before enabling live execution.


## WebSocket timeout note

`WebSocketTimeoutException: Connection timed out` can happen on idle feeds. The bot now treats this as expected, sends heartbeat pings, and reconnects automatically when needed. Tune `WS_RECV_TIMEOUT_SECONDS`, `WS_HEARTBEAT_SECONDS`, and `WS_RECONNECT_SECONDS` in `.env` for your network conditions.


## Console/UI quality-of-life

- The bot prints a P&L tracker line every evaluation cycle.
- Every `DASHBOARD_INTERVAL_SECONDS`, it prints a compact dashboard block (cycle/time/mode/markets/signals/open spreads/P&L).
- On Windows, the terminal title is updated with live status (`ENABLE_WINDOWS_TITLE=true`).
- Trade actions are highlighted with clear markers (`🚨 TRADE ENTERED`, `✅ TRADE EXITED`).
- Set `CLEAR_CONSOLE_SECONDS=60` (or any value) to routinely clear console output; set `0` to disable.
