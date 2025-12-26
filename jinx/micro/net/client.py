"""Advanced OpenAI client management with connection pooling and retry logic."""

from __future__ import annotations

import os
import threading
from urllib.parse import urlparse
from jinx.bootstrap import ensure_optional
import importlib
from typing import Any, Optional
from dataclasses import dataclass
import time

openai = ensure_optional(["openai"])["openai"]  # dynamic import


@dataclass
class ClientMetrics:
    """Track OpenAI client usage metrics."""
    requests_count: int = 0
    errors_count: int = 0
    last_request_time: float = 0.0
    creation_time: float = 0.0


# Thread-safe singleton with lazy initialization
_cortex: Any | None = None
_cortex_lock = threading.RLock()
_metrics = ClientMetrics()


def _pick_proxy_env() -> str | None:
    # Preference order: explicit PROXY, then HTTPS_PROXY, then HTTP_PROXY (case-insensitive)
    for key in ("PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.getenv(key)
        if val:
            return val
    return None


def get_openai_client() -> Any:
    """Return a thread-safe singleton OpenAI client with connection pooling.

    Features:
    - Thread-safe lazy initialization
    - Automatic proxy detection (SOCKS/HTTP/HTTPS)
    - Connection pooling and keepalive
    - Metrics tracking
    - Graceful fallback on configuration errors

    Supports both SOCKS and HTTP(S) proxies by constructing an httpx.Client
    with the appropriate transport or proxies mapping.
    """
    global _cortex, _metrics
    
    # Double-checked locking pattern for thread safety
    if _cortex is not None:
        return _cortex
    
    with _cortex_lock:
        # Check again inside lock
        if _cortex is not None:
            return _cortex
        
        _metrics.creation_time = time.time()
        proxy = _pick_proxy_env()

        # If OpenAI SDK isn't installed, fail fast with a clear error.
        try:
            if bool(getattr(openai, "__jinx_optional_missing__", False)):
                raise RuntimeError("Optional dependency missing: openai")
        except Exception:
            raise RuntimeError("Optional dependency missing: openai")
        
        # Configure httpx client with optimized settings
        try:
            httpx = importlib.import_module("httpx")
        except ImportError:
            raise RuntimeError("Optional dependency missing: httpx")
        
        # Advanced httpx configuration
        client_config = {
            "timeout": httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=10.0,
                pool=5.0
            ),
            "limits": httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0
            ),
        }
        
        if proxy:
            try:
                scheme = (urlparse(proxy).scheme or "").lower()
                if scheme.startswith("socks"):
                    # SOCKS proxy requires httpx-socks
                    try:
                        httpx_socks = importlib.import_module("httpx_socks")
                    except ImportError:
                        raise RuntimeError("Optional dependency missing: httpx_socks")
                    
                    transport = httpx_socks.SyncProxyTransport.from_url(proxy)
                    client_config["transport"] = transport
                    http_client = httpx.Client(**client_config)
                else:
                    # HTTP(S) proxy via httpx native proxies support
                    client_config["proxies"] = proxy
                    http_client = httpx.Client(**client_config)
                
                _cortex = openai.OpenAI(http_client=http_client)
            except Exception as e:
                # Fallback to direct client if proxy configuration fails
                try:
                    # Log proxy failure for debugging
                    import sys
                    print(f"Warning: Proxy configuration failed ({e}), using direct connection", file=sys.stderr)
                except Exception:
                    pass
                _cortex = openai.OpenAI(http_client=httpx.Client(**client_config))
        else:
            # No proxy - use direct connection with optimized settings
            _cortex = openai.OpenAI(http_client=httpx.Client(**client_config))
        
        return _cortex


def prewarm_openai_client() -> None:
    """Instantiate the OpenAI client early to warm HTTP pool/proxy resolution.

    Safe to call multiple times; returns immediately if already initialized.
    Thread-safe and non-blocking.
    """
    try:
        _ = get_openai_client()
    except Exception:
        # Best-effort: swallow errors — prewarm should never crash startup
        pass


def get_client_metrics() -> ClientMetrics:
    """Return current client usage metrics."""
    return _metrics


def track_request(success: bool = True) -> None:
    """Track API request for metrics."""
    global _metrics
    with _cortex_lock:
        _metrics.requests_count += 1
        _metrics.last_request_time = time.time()
        if not success:
            _metrics.errors_count += 1


def reset_client() -> None:
    """Force client reset (for testing or error recovery)."""
    global _cortex, _metrics
    with _cortex_lock:
        if _cortex is not None:
            try:
                # Close existing client connections
                if hasattr(_cortex, '_client') and hasattr(_cortex._client, 'close'):
                    _cortex._client.close()
            except Exception:
                pass
        _cortex = None
        _metrics = ClientMetrics()
