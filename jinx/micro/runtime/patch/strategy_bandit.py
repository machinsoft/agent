from __future__ import annotations

import os
import json
import time
from typing import Dict, Tuple, List

_BANDIT_PATH = os.path.join(".jinx", "brain", "patch_bandit.json")
_METRICS_PATH = os.path.join(".jinx", "stats", "patch_metrics.json")
_OS_MAKEDIRS = os.makedirs


def _ensure_dir() -> None:
    try:
        _OS_MAKEDIRS(os.path.dirname(_BANDIT_PATH), exist_ok=True)
    except Exception:
        pass


def _now() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0


def _load() -> Dict[str, Dict[str, Dict[str, float]]]:
    try:
        with open(_BANDIT_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj  # type: ignore[return-value]
    except Exception:
        pass
    return {}


def _load_metrics() -> Dict:
    try:
        with open(_METRICS_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    return {}


def _save(obj: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    _ensure_dir()
    try:
        with open(_BANDIT_PATH, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass


def _decay(value: float, last_ts: float, half_life_sec: float) -> float:
    try:
        dt = max(0.0, _now() - float(last_ts or 0.0))
        if half_life_sec <= 0.0:
            return value
        return value * (0.5 ** (dt / half_life_sec))
    except Exception:
        return value


def bandit_order_for_context(ctx: str, strategies: List[str]) -> List[str]:
    """Return strategies ordered by UCB-like score for a given context.

    Keeps exploration by giving unseen strategies a modest prior.
    """
    data = _load()
    metrics = _load_metrics()
    sdata = data.get(ctx) or {}
    now = _now()
    hl = 1800.0
    out: List[Tuple[float, str]] = []
    # Global trial count for exploration term
    total_trials = 1.0
    for s in strategies:
        ent = sdata.get(s) or {"succ": 0.0, "fail": 0.0, "ts": now}
        total_trials += ent.get("succ", 0.0) + ent.get("fail", 0.0)
    for s in strategies:
        ent = sdata.get(s) or {"succ": 0.0, "fail": 0.0, "ts": now}
        succ = _decay(float(ent.get("succ", 0.0)), float(ent.get("ts", now)), hl)
        fail = _decay(float(ent.get("fail", 0.0)), float(ent.get("ts", now)), hl)
        trials = max(1.0, succ + fail)
        rate = succ / trials
        # UCB-like exploration bonus
        import math
        bonus = math.sqrt(max(0.0, math.log(total_trials) / trials))
        # Metrics penalty: prefer strategies with lower fail rate and smaller diffs in this context
        pen = 0.0
        try:
            c = (metrics.get("contexts") or {}).get(ctx or "") or {}
            sroot = (c.get("strategies") or {})
            sm = sroot.get(s) or {}
            scount = int(sm.get("count", 0))
            sfail = int(sm.get("fail", 0))
            sdiff = float(sm.get("avg_diff", 0.0) or 0.0)
            fail_rate = (sfail / max(1, scount)) if scount > 0 else 0.0
            diff_norm = min(1.0, sdiff / 200.0)
            pen = 0.4 * fail_rate + 0.2 * diff_norm
            # light penalty for riskier families
            base = s
            for key in ("_wide", "_0.64", "_0.55"):
                base = base.replace(key, "")
            if base in ("write", "search_line"):
                pen += 0.1
        except Exception:
            pen = 0.0
        # Family priors (tiny nudges toward safer/more precise strategies)
        base = s
        # normalize prefixes
        base = base.replace("search_semantic", "semantic")
        base = base.replace("cg_window_search_", "cg_window_")
        for key in ("_wide", "_0.64", "_0.55"):
            base = base.replace(key, "")
        priors = {
            "cg_scope": 0.12,
            "symbol": 0.10,
            "cg_window_def": 0.08,
            "cg_window_caller": 0.06,
            "cg_window_callee_def": 0.06,
            "semantic": 0.05,
            "context": 0.03,
            "anchor": 0.02,
            "ts_line": 0.01,
        }
        prior = 0.0
        for fam, w in priors.items():
            if base.startswith(fam):
                prior = w
                break
        score = rate + 0.4 * bonus + prior - pen
        out.append((score, s))
    out.sort(key=lambda t: t[0], reverse=True)
    return [s for _sc, s in out]


def bandit_update(ctx: str, strategy: str, success: bool) -> None:
    data = _load()
    now = _now()
    sdata = data.setdefault(ctx, {})
    ent = sdata.get(strategy) or {"succ": 0.0, "fail": 0.0, "ts": now}
    # Decay existing counts to avoid stale history domination
    hl = 1800.0
    ent["succ"] = _decay(float(ent.get("succ", 0.0)), float(ent.get("ts", now)), hl)
    ent["fail"] = _decay(float(ent.get("fail", 0.0)), float(ent.get("ts", now)), hl)
    if success:
        ent["succ"] = float(ent.get("succ", 0.0)) + 1.0
    else:
        ent["fail"] = float(ent.get("fail", 0.0)) + 1.0
    ent["ts"] = now
    sdata[strategy] = ent
    data[ctx] = sdata
    _save(data)
