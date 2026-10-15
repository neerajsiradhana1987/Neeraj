"""Configuration loading, defaults and environment overrides."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    base_url: str = "https://openrouter.ai/api/v1"   # any OpenAI-compatible endpoint
    api_key: str = ""                                # never store keys in config.json
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 500
    timeout_seconds: int = 45
    enabled: bool = True


@dataclass
class RiskConfig:
    max_position_pct: float = 0.15
    max_positions: int = 5
    risk_per_trade_pct: float = 0.01
    max_daily_loss_pct: float = 0.03
    min_cash_pct: float = 0.10
    max_order_value: float = 200000.0
    cooldown_minutes: int = 60
    atr_stop_multiplier: float = 2.0
    reward_risk_ratio: float = 2.0


@dataclass
class BrokerConfig:
    starting_cash: float = 500000.0     # INR, only used for a fresh state file
    slippage_bps: float = 5.0           # adverse slippage on every fill
    allow_short: bool = False
    brokerage_flat: float = 20.0
    brokerage_pct: float = 0.0003
    stt_pct_sell: float = 0.001
    exchange_txn_pct: float = 0.0000297
    sebi_pct: float = 0.000001
    stamp_duty_pct_buy: float = 0.00015
    gst_pct: float = 0.18


@dataclass
class MarketConfig:
    symbols: list = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    ])
    interval: str = "5m"
    bars: int = 200
    session_only: bool = True
    holidays: list = field(default_factory=list)
    staleness_multiplier: float = 4.0


@dataclass
class DataConfig:
    source: str = "yahoo"     # yahoo | csv | synthetic
    csv_directory: str = "data"
    seed: int = 1234


@dataclass
class IndicesConfig:
    """Defaults for the ``indices`` command (trend / range / reversal / options)."""

    names: list = field(default_factory=lambda: ["NIFTY", "BANKNIFTY", "SENSEX"])
    interval: str = "5m"
    bars: int = 200
    option_source: str = "nse"     # nse | synthetic | off
    ladder_strikes: int = 25       # strikes kept in JSON output around the ATM strike
    data_source: str = ""          # optional override of data.source for index bars
    refresh_seconds: int = 60      # minimum age before the dashboard refetches the board


@dataclass
class AgentConfig:
    state_dir: str = "state"
    poll_seconds: int = 300
    llm: LLMConfig = field(default_factory=LLMConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    data: DataConfig = field(default_factory=DataConfig)
    indices: IndicesConfig = field(default_factory=IndicesConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def redacted_dict(self) -> dict:
        data = self.to_dict()
        if data["llm"].get("api_key"):
            data["llm"]["api_key"] = "***"
        return data


def _merge(section_data: dict, section_cls):
    """Return an instance of ``section_cls`` with defaults filled from ``section_data``."""
    defaults = section_cls()
    unknown = set(section_data) - set(asdict(defaults))
    if unknown:
        raise ValueError(f"unknown keys for {section_cls.__name__}: {sorted(unknown)}")
    merged = {**asdict(defaults), **{k: v for k, v in section_data.items() if v is not None}}
    return section_cls(**merged)


def load_dotenv(path: str = ".env") -> dict:
    values = {}
    env_path = Path(path)
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config(path: str = "config.json", dotenv: str = ".env") -> AgentConfig:
    """Load configuration, layering env vars on top of the JSON file."""
    for key, value in load_dotenv(dotenv).items():
        os.environ.setdefault(key, value)

    raw = {}
    config_path = Path(path)
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")

    config = AgentConfig(
        state_dir=raw.get("state_dir", "state"),
        poll_seconds=int(raw.get("poll_seconds", 300)),
        llm=_merge(raw.get("llm", {}), LLMConfig),
        risk=_merge(raw.get("risk", {}), RiskConfig),
        broker=_merge(raw.get("broker", {}), BrokerConfig),
        market=_merge(raw.get("market", {}), MarketConfig),
        data=_merge(raw.get("data", {}), DataConfig),
        indices=_merge(raw.get("indices", {}), IndicesConfig),
    )

    # Environment overrides (secrets and deployment-specific values live here).
    env = os.environ
    if env.get("TRADING_LLM_API_KEY"):
        config.llm.api_key = env["TRADING_LLM_API_KEY"].strip()
    if env.get("TRADING_LLM_BASE_URL"):
        config.llm.base_url = env["TRADING_LLM_BASE_URL"].strip().rstrip("/")
    if env.get("TRADING_LLM_MODEL"):
        config.llm.model = env["TRADING_LLM_MODEL"].strip()
    if env.get("TRADING_LLM_ENABLED"):
        config.llm.enabled = env["TRADING_LLM_ENABLED"].strip().lower() in ("1", "true", "yes")
    if env.get("TRADING_INDICES_OPTION_SOURCE"):
        config.indices.option_source = env["TRADING_INDICES_OPTION_SOURCE"].strip().lower()
    if env.get("TRADING_INDICES_INTERVAL"):
        config.indices.interval = env["TRADING_INDICES_INTERVAL"].strip()
    if not config.llm.api_key:
        config.llm.enabled = False
    return config


def write_default_config(path: str = "config.json", force: bool = False) -> Path:
    config_path = Path(path)
    if config_path.exists() and not force:
        raise FileExistsError(f"{path} already exists (use --force to overwrite)")
    defaults = AgentConfig().redacted_dict()
    config_path.write_text(json.dumps(defaults, indent=2) + "\n", encoding="utf-8")
    return config_path


DOTENV_EXAMPLE = """# Copy to .env and fill in your key. Never commit the real .env.
# Any OpenAI-compatible provider works.
TRADING_LLM_API_KEY=
TRADING_LLM_BASE_URL=https://openrouter.ai/api/v1
TRADING_LLM_MODEL=openai/gpt-4o-mini
"""


def write_dotenv_example(path: str = ".env.example", force: bool = False) -> Path:
    env_path = Path(path)
    if env_path.exists() and not force:
        return env_path
    env_path.write_text(DOTENV_EXAMPLE, encoding="utf-8")
    return env_path
