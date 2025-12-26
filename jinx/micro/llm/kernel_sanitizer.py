from __future__ import annotations

import re
from typing import Optional, List, Pattern
from functools import lru_cache
from dataclasses import dataclass
import threading


@dataclass
class SanitizerRule:
    """Configurable code sanitization rule."""
    name: str
    pattern: str
    severity: str = "high"  # high, medium, low
    enabled: bool = True


class KernelSanitizer:
    """Advanced kernel code sanitization with extensible rules."""
    
    _instance: 'KernelSanitizer | None' = None
    _lock = threading.RLock()
    
    def __init__(self):
        self._patterns: List[Pattern[str]] = []
        self._triple_quote_markers = ("'''", '"""')
        self._setup_default_rules()
    
    @classmethod
    def get_instance(cls) -> 'KernelSanitizer':
        """Thread-safe singleton access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _setup_default_rules(self) -> None:
        """Initialize default forbidden patterns."""
        default_rules = [
            SanitizerRule("eval", r"\beval\s*\(", "high"),
            SanitizerRule("exec", r"\bexec\s*\(", "high"),
            SanitizerRule("sys_exit", r"sys\.exit\b", "high"),
            SanitizerRule("os_exit", r"os\._exit\b", "high"),
            SanitizerRule("subprocess_popen", r"subprocess\.Popen\b", "high"),
            SanitizerRule("subprocess_shell", r"subprocess\.run\b.*shell\s*=\s*True", "high"),
            SanitizerRule("pip_install", r"pip\b", "medium"),
            SanitizerRule("os_system", r"os\.system\b", "high"),
            SanitizerRule("import_module", r"importlib\.import_module\b", "medium"),
        ]
        
        for rule in default_rules:
            self._add_rule(rule)
    
    def _add_rule(self, rule: SanitizerRule) -> None:
        """Add a sanitization rule with compilation."""
        if not rule.enabled:
            return
        
        try:
            compiled = re.compile(rule.pattern, re.IGNORECASE)
            self._patterns.append(compiled)
        except re.error:
            # Skip invalid patterns
            pass
    
    def add_custom_rule(self, rule: SanitizerRule) -> None:
        """Add custom sanitization rule at runtime."""
        with self._lock:
            self._add_rule(rule)
    
    @lru_cache(maxsize=512)
    def check(self, code: str) -> bool:
        """Check if code passes sanitization rules (True = safe, False = forbidden)."""
        if not code:
            return False
        
        # Check for triple quotes
        if any(marker in code for marker in self._triple_quote_markers):
            return False
        
        # Check forbidden patterns
        for pattern in self._patterns:
            if pattern.search(code):
                return False
        
        return True


# Global sanitizer instance
_sanitizer = KernelSanitizer.get_instance()

# Legacy compatibility
_FORBIDDEN_PATTERNS = [pat.pattern for pat in _sanitizer._patterns]
_DEF_TRIPLE = ("'''", '"""')
_KERNEL_MAXCHARS = 3000


def sanitize_kernels(code: str) -> str:
    """Return code if it passes basic safety/size checks; else return empty string.

    Rules:
    - Forbid triple quotes.
    - Enforce max char length.
    - Reject if obvious forbidden tokens are present.
    - This is a best-effort hygiene gate; not a sandbox replacement.
    """
    body = (code or "").strip()
    if not body:
        return ""
    
    max_chars = _KERNEL_MAXCHARS
    
    if len(body) > max_chars:
        return ""
    
    # Use advanced sanitizer
    if not _sanitizer.check(body):
        return ""
    
    return body
