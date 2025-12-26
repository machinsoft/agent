from __future__ import annotations

"""
Centralized runtime settings for Jinx.

Single source of truth for configuration. Values are resolved from environment
variables with safe defaults. No hard dependency on .env files; loading those is
handled by jinx.bootstrap.load_env().

Design principles:
- Pure dataclasses (no heavy frameworks) for portability and reliability.
- Explicit parsing helpers for booleans/CSV to avoid ambiguity.
- Readable stdout dumper for operational clarity (no magic). 
- Backwards compatible with existing env keys used across micro-modules.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import os
import json
from jinx.micro.common.config import clamp_int, clamp_float

# Reuse env keys from micro modules where applicable
ENV_OPENAI_VECTOR_STORE_ID = "OPENAI_VECTOR_STORE_ID"
ENV_OPENAI_FORCE_FILE_SEARCH = "OPENAI_FORCE_FILE_SEARCH"


def _is_on(val: Optional[str]) -> bool:
    return (val or "0").strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _auto_threads() -> int:
    cpus = os.cpu_count() or 4
    # Favor more threads for I/O bound workloads but keep an upper cap
    return max(4, min(32, cpus * 4))


def _auto_queue_maxsize() -> int:
    cpus = os.cpu_count() or 1
    # Scale modestly with cores to avoid memory blowups
    return max(100, min(2000, 100 + cpus * 50))


def _auto_rt_budget_ms() -> int:
    cpus = os.cpu_count() or 1
    # Conservative default budget; slightly lower with many cores
    return 30 if cpus >= 8 else 40


@dataclass(slots=True)
class RuntimeSettings:
    queue_maxsize: int = 100
    use_priority_queue: bool = False
    queue_policy: str = "block"  # future: drop_newest, block
    supervise_tasks: bool = True
    autorestart_limit: int = 5
    backoff_min_ms: int = 50
    backoff_max_ms: int = 2000
    hard_rt_budget_ms: int = 40  # soft budget for cooperative sections
    threads_max_workers: int = 8
    auto_tune: bool = True
    saturate_enable_ratio: float = 0.6
    saturate_disable_ratio: float = 0.25
    saturate_window_ms: int = 500


@dataclass(slots=True)
class OpenAISettings:
    api_key: Optional[str] = None
    model: str = "gpt-5"
    proxy: Optional[str] = None
    vector_store_ids: List[str] = field(default_factory=list)
    force_file_search: bool = True


@dataclass(slots=True)
class Settings:
    pulse: int = 100
    timeout: int = 30
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @staticmethod
    def from_env(overrides: Optional[Dict[str, Any]] = None) -> "Settings":
        o: Dict[str, Any] = overrides or {}
        # Base values
        pulse_raw = int(o.get("pulse", 100))
        timeout_raw = int(o.get("timeout", 30))
        s = Settings(
            pulse=clamp_int(pulse_raw, 20, 1000),
            # Allow large idle timeouts (up to 24h) to support long sessions
            timeout=clamp_int(timeout_raw, 5, 86400),
        )
        # OpenAI
        s.openai.api_key = str(o.get("openai_api_key", "") or "") or None
        s.openai.model = str(o.get("openai_model", s.openai.model))
        s.openai.proxy = str(o.get("proxy", "") or "") or None
        s.openai.vector_store_ids = _parse_csv(
            str(o.get("vector_store_ids", ""))
        )
        s.openai.force_file_search = bool(
            _is_on(str(o.get("force_file_search", "1")))
        )
        # Runtime
        rt = s.runtime
        rt.queue_maxsize = clamp_int(int(o.get("queue_maxsize", _auto_queue_maxsize())), 50, 10000)
        rt.use_priority_queue = bool(
            _is_on(str(o.get("use_priority_queue", "1")))
        )
        rt.queue_policy = str(o.get("queue_policy", rt.queue_policy))
        rt.supervise_tasks = not _is_on(str(o.get("no_supervisor", "0")))
        rt.autorestart_limit = clamp_int(int(o.get("autorestart_limit", rt.autorestart_limit)), 0, 50)
        rt.backoff_min_ms = clamp_int(int(o.get("backoff_min_ms", rt.backoff_min_ms)), 10, 10000)
        rt.backoff_max_ms = clamp_int(int(o.get("backoff_max_ms", rt.backoff_max_ms)), rt.backoff_min_ms, 60000)
        rt.hard_rt_budget_ms = clamp_int(int(o.get("hard_rt_budget_ms", _auto_rt_budget_ms())), 10, 200)
        rt.threads_max_workers = clamp_int(int(o.get("threads", _auto_threads())), 2, 64)
        rt.auto_tune = not _is_on(str(o.get("no_autotune", "0")))
        try:
            rt.saturate_enable_ratio = clamp_float(float(o.get("saturate_enable_ratio", rt.saturate_enable_ratio)) , 0.0, 1.0)
            rt.saturate_disable_ratio = clamp_float(float(o.get("saturate_disable_ratio", rt.saturate_disable_ratio)) , 0.0, 1.0)
            rt.saturate_window_ms = clamp_int(int(o.get("saturate_window_ms", rt.saturate_window_ms)), 50, 10000)
        except Exception:
            pass
        return s

    # Operational helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def print_stdout(self) -> None:
        # Subtle Jinx vibe without breaking ops readability
        header = "‖ Jinx Settings — tuned wires, humming neon \u2728"
        print(header)
        print(self.to_json())

    # Apply critical settings to global state for backward compatibility
    def apply_to_state(self) -> None:
        try:
            import jinx.state as jx_state
            jx_state.pulse = int(self.pulse)
            jx_state.boom_limit = int(self.timeout)
        except Exception:
            # Do not crash on settings apply; keep resilient
            pass
