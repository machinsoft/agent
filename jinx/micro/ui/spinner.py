from __future__ import annotations

"""Spinner micro-module.

Renders a non-blocking terminal spinner while background tasks are running.
The spinner terminates when the provided event is set.
"""

import asyncio
import time
import sys
import random
import importlib
import os

from jinx.spinner.phrases import PHRASES as phrases
from jinx.spinner import ascii_mode as _ascii_mode, can_render as _can_render
from jinx.spinner import get_spinner_frames, get_hearts
import jinx.state as state
from .spinner_util import format_activity_detail, parse_env_bool, parse_env_int
from jinx.micro.rt.backpressure import update_from_lag, clear_throttle_if_ttl


def _ptk_symbols():
    """Best-effort prompt_toolkit symbols.

    Returns (print_formatted_text, FormattedText, patch_stdout) or (None, None, None).
    """
    try:
        ptk = importlib.import_module("prompt_toolkit")
        ptk_fmt = importlib.import_module("prompt_toolkit.formatted_text")
        ptk_patch = importlib.import_module("prompt_toolkit.patch_stdout")
        return (
            getattr(ptk, "print_formatted_text"),
            getattr(ptk_fmt, "FormattedText"),
            getattr(ptk_patch, "patch_stdout"),
        )
    except Exception:
        return (None, None, None)


async def sigil_spin(evt: asyncio.Event) -> None:
    """Minimal, pretty spinner that shows pulse and spins until evt is set.

    Parameters
    ----------
    evt : asyncio.Event
        Event signaling spinner shutdown.
    """
    try:
        _mode_env = os.getenv("JINX_SPINNER_MODE")
    except Exception:
        _mode_env = None
    if _mode_env is None:
        # If prompt_toolkit UI is active, prefer toolbar (advanced, non-intrusive).
        # Otherwise on Windows prefer a visible single-line spinner.
        try:
            if bool(getattr(state, "ui_prompt_toolkit", False)):
                mode = "toolbar"
            else:
                mode = "line" if sys.platform.startswith("win") else "toolbar"
        except Exception:
            mode = "line" if sys.platform.startswith("win") else "toolbar"
    else:
        mode = (_mode_env.strip().lower() or "toolbar")
    # If toolbar was requested but prompt_toolkit UI isn't active, fall back to visible mode.
    try:
        if mode == "toolbar" and not bool(getattr(state, "ui_prompt_toolkit", False)):
            mode = "line"
    except Exception:
        pass
    if mode == "toolbar":
        # Toolbar mode: do not print; let PromptSession bottom_toolbar render from state
        try:
            state.spin_on = True
            state.spin_t0 = float(time.perf_counter())
            last_tick = time.perf_counter()
            lag_ema_ms = 0.0
            alpha = 0.3
            hi_cnt = 0
            lo_cnt = 0
            while not evt.is_set():
                now = time.perf_counter()
                gap = now - last_tick
                last_tick = now
                # Nominal interval ~0.08s
                over_ms = max(0.0, (gap - 0.08) * 1000.0)
                lag_ema_ms = alpha * over_ms + (1.0 - alpha) * lag_ema_ms
                # Expose lag to state for diagnostics/autotune
                try:
                    state.lag_ema_ms = float(lag_ema_ms)
                except Exception:
                    pass
                # Hysteresis via backpressure helpers (TTL-aware)
                try:
                    clear_throttle_if_ttl(now)
                    update_from_lag(lag_ema_ms, hi_ms=120.0, lo_ms=50.0)
                except Exception:
                    pass
                await asyncio.sleep(0.08)
        finally:
            state.spin_on = False
        return

    if mode == "line":
        # Line mode: advanced single-line spinner that doesn't spam output.
        # Designed to coexist with simple input by updating a single carriage-return line.
        try:
            state.spin_on = True
            state.spin_t0 = float(time.perf_counter())
            # Dynamic phrase pool with fallback
            try:
                _phr = list(phrases) if phrases else ["thinking"]
            except Exception:
                _phr = ["thinking"]
            phrase_idx = 0
            last_phrase = 0.0
            # Spinner frames
            frames = ["|", "/", "-", "\\"]
            fidx = 0
            # Lag EMA
            last_tick = time.perf_counter()
            lag_ema_ms = 0.0
            alpha = 0.3
            last_emit = -1.0
            while not evt.is_set():
                now = time.perf_counter()
                # update phrase ~ every 0.9s
                if (now - last_phrase) >= 0.9:
                    try:
                        if _phr:
                            phrase_idx = random.randrange(len(_phr))
                    except Exception:
                        phrase_idx = (phrase_idx + 1) % max(1, len(_phr))
                    last_phrase = now

                # lag estimate (tick ~0.08s)
                gap = now - last_tick
                last_tick = now
                over_ms = max(0.0, (gap - 0.08) * 1000.0)
                lag_ema_ms = alpha * over_ms + (1.0 - alpha) * lag_ema_ms
                try:
                    state.lag_ema_ms = float(lag_ema_ms)
                except Exception:
                    pass

                if (now - last_emit) >= 0.25:
                    last_emit = now
                    try:
                        pulse = int(getattr(state, "pulse", 0) or 0)
                    except Exception:
                        pulse = 0
                    try:
                        act = (getattr(state, "activity", "") or "").strip() or "thinking"
                    except Exception:
                        act = "thinking"
                    try:
                        age = 0.0
                        ts = float(getattr(state, "activity_ts", 0.0) or 0.0)
                        if ts:
                            age = max(0.0, time.perf_counter() - ts)
                    except Exception:
                        age = 0.0
                    # detail (compact)
                    det_str = ""
                    try:
                        det = getattr(state, "activity_detail", None)
                        det_str2, _stage, _tasks = format_activity_detail(det)
                        det_str = det_str2
                    except Exception:
                        det_str = ""
                    try:
                        fr = frames[fidx % len(frames)]
                        fidx += 1
                        phrase = _phr[phrase_idx] if _phr else "thinking"
                        # compact lag string (only if meaningful)
                        lag_str = f" lag:{lag_ema_ms:.0f}ms" if lag_ema_ms >= 40.0 else ""
                        line = f"[Jinx] {fr} ❤{pulse} {phrase} | {act} [{age:.1f}s]{det_str}{lag_str}"
                        # Do not overwrite the user's typing in simple input mode
                        try:
                            if bool(getattr(state, "ui_input_active", False)):
                                raise RuntimeError("ui_input_active")
                        except Exception:
                            pass
                        else:
                            # Carriage-return overwrite (do not spam new lines)
                            sys.stdout.write("\r" + line + " " * 10)
                            sys.stdout.flush()
                    except Exception:
                        pass
                await asyncio.sleep(0.08)
        finally:
            state.spin_on = False
            try:
                sys.stdout.write("\r" + (" " * 120) + "\r")
                sys.stdout.flush()
            except Exception:
                pass
        return

    # Lazy import with auto-install of prompt_toolkit
    print_formatted_text, FormattedText, patch_stdout = _ptk_symbols()
    if print_formatted_text is None or FormattedText is None or patch_stdout is None:
        # No prompt_toolkit available: keep a minimal heartbeat without printing.
        try:
            state.spin_on = True
            state.spin_t0 = float(time.perf_counter())
            while not evt.is_set():
                await asyncio.sleep(0.08)
        finally:
            state.spin_on = False
        return

    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    fx = print_formatted_text
    ft = FormattedText
    t0 = time.perf_counter()
    ascii_mode = _ascii_mode()
    heart_a, heart_b = get_hearts(ascii_mode, can=lambda s: _can_render(s, enc))
    phrase_idx = 0
    last_change = 0.0  # seconds since t0 when phrase last changed
    # Thin circular spinner frames (Unicode) with ASCII fallback
    spin_frames = get_spinner_frames(ascii_mode, can=lambda s: _can_render(s, enc))
    # Event loop lag EMA tracking
    last_tick = time.perf_counter()
    lag_ema_ms = 0.0
    alpha = 0.3
    # Redraw throttling and change detection
    last_emit = t0
    last_stage = None
    last_tasks = None
    last_line = ""

    # Ensure spinner writes cooperate with the active prompt by patching stdout
    try:
        state.spin_on = True
        state.spin_t0 = float(time.perf_counter())
        with patch_stdout(raw=True):
            while not evt.is_set():
                now = time.perf_counter()
                dt = now - t0
                pulse = state.pulse
                clr = "ansibrightgreen"
                # Change phrase a bit slower (~every 0.85s)
                if (dt - last_change) >= 0.85:
                    new_idx = random.randrange(len(phrases)) if phrases else 0
                    if phrases and len(phrases) > 1 and new_idx == phrase_idx:
                        new_idx = (new_idx + 1) % len(phrases)
                    phrase_idx = new_idx
                    last_change = dt
                phrase = phrases[phrase_idx]

                # Loading dots cadence (0..3 dots cycling)
                n = int(dt * 0.8) % 4
                dd = "." * n

                # Pulsating heart (toggle ~1.5 Hz) with minimal size change
                beat = int(dt * 1.5) % 2
                heart = heart_a if beat == 0 else heart_b
                style = clr if beat == 0 else f"{clr} bold"

                # ASCII spinner right after pulse (~10–12 FPS)
                sidx = int(dt * 10) % len(spin_frames)
                spin = spin_frames[sidx]

                # Compose dynamic activity description
                show_act = parse_env_bool("JINX_SPINNER_ACTIVITY", True)
                desc = ""
                if show_act:
                    try:
                        act = (getattr(state, "activity", "") or "").strip()
                        if act:
                            age = 0.0
                            try:
                                age = max(0.0, time.perf_counter() - float(getattr(state, "activity_ts", 0.0) or 0.0))
                            except Exception:
                                age = 0.0
                            desc = f" | {act} [{age:.1f}s]"
                    except Exception:
                        desc = ""

                # Compose compact detail from structured activity_detail if enabled
                show_det = parse_env_bool("JINX_SPINNER_ACTIVITY_DETAIL", True)
                det_str = ""
                stage = None
                tasks = None
                if show_det:
                    try:
                        det = getattr(state, "activity_detail", None)
                        det_str2, stage2, tasks2 = format_activity_detail(det)
                        det_str = det_str2
                        if stage2 is not None:
                            stage = stage2
                        if tasks2 is not None:
                            tasks = tasks2
                    except Exception:
                        det_str = ""

                # Compute event loop lag EMA (approximate): expected ~0.06s per tick
                gap = now - last_tick
                last_tick = now
                show_lag = parse_env_bool("JINX_SPINNER_SHOW_LAG", False)
                if show_lag:
                    # Only accumulate positive overruns over nominal interval
                    over_ms = max(0.0, (gap - 0.06) * 1000.0)
                    lag_ema_ms = alpha * over_ms + (1.0 - alpha) * lag_ema_ms
                # Expose lag EMA and color adapt based on stage/lag (optional)
                try:
                    try:
                        state.lag_ema_ms = float(lag_ema_ms)
                    except Exception:
                        pass
                    if stage and str(stage).startswith("repair"):
                        clr = "ansibrightcyan"
                    elif show_lag and lag_ema_ms > 120.0:
                        clr = "ansired"
                    elif show_lag and lag_ema_ms > 50.0:
                        clr = "ansiyellow"
                    else:
                        clr = "ansibrightgreen"
                except Exception:
                    clr = "ansibrightgreen"
                # Lag-driven throttle hysteresis in print mode as well
                try:
                    clear_throttle_if_ttl(now)
                    update_from_lag(lag_ema_ms, hi_ms=120.0, lo_ms=50.0)
                except Exception:
                    pass
                # Build single-line render with optional lag
                lag_str = (f" lag:{lag_ema_ms:.1f}ms" if show_lag else "")
                line = f"{heart} {pulse} {spin} {dd} {phrase}{desc}{det_str}{lag_str} (total {dt:.1f}s)"

                # Redraw throttling: only when content changes significantly or interval elapsed
                min_ms = parse_env_int("JINX_SPINNER_MIN_UPDATE_MS", 160)
                redraw_on_change = parse_env_bool("JINX_SPINNER_REDRAW_ONLY_ON_CHANGE", True)

                significant_change = (stage is not None and stage != last_stage) or (tasks is not None and tasks != last_tasks)
                time_elapsed = (now - last_emit) * 1000.0 >= float(min_ms)
                content_changed = (line != last_line)

                if (not redraw_on_change and time_elapsed) or (redraw_on_change and (significant_change or (time_elapsed and content_changed))):
                    fx(ft([(style, line)]), end="\r", flush=False)
                    last_emit = now
                    last_line = line
                    last_stage = stage if stage is not None else last_stage
                    last_tasks = tasks if tasks is not None else last_tasks

                await asyncio.sleep(0.06)
    finally:
        state.spin_on = False
        
    
    fx(ft([("", " " * 80)]), end="\r", flush=True)


__all__ = [
    "sigil_spin",
]
