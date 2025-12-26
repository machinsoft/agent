"""Interactive input micro-module.

Provides an async prompt using prompt_toolkit and pushes sanitized user input
into an asyncio queue. Includes an inactivity watchdog that emits
"<no_response>" after a configurable timeout to keep the agent responsive.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import time
import contextlib
import sys
from jinx.micro.runtime.msg_id import ensure_message_id as _ensure_id
from typing import Any, cast
from jinx.logging_service import blast_mem, bomb_log
from jinx.log_paths import TRIGGER_ECHOES, BLUE_WHISPERS
from jinx.async_utils.queue import try_put_nowait
import jinx.state as jx_state
from jinx.micro.ui.spinner_util import format_activity_detail, parse_env_bool, parse_env_int
from jinx.micro.rt.backpressure import set_throttle_ttl





async def _simple_input_loop(qe: asyncio.Queue[str]) -> None:
    """Simple input loop fallback for Windows PowerShell when prompt_toolkit fails."""
    try:
        jx_state.ui_prompt_toolkit = False
    except Exception:
        pass
    print("\nJinx ready (simple input mode)")
    print("Type your requests and press Enter\n")
    
    while True:
        # Check for shutdown FIRST
        if jx_state.shutdown_event.is_set():
            break
        
        try:
            # Get input from user
            try:
                loop = asyncio.get_running_loop()
                # Prevent spinner from overwriting the user's typing
                try:
                    jx_state.ui_input_active = True
                except Exception:
                    pass
                try:
                    # Best-effort: clear any in-place spinner line before showing prompt
                    try:
                        import sys as _sys
                        _sys.stdout.write("\r" + (" " * 140) + "\r")
                        _sys.stdout.flush()
                    except Exception:
                        pass
                    user_input = await loop.run_in_executor(None, lambda: input("> "))
                finally:
                    try:
                        jx_state.ui_input_active = False
                    except Exception:
                        pass
                
                if user_input.strip():
                    # Try to log, but don't fail if logging fails
                    try:
                        await bomb_log(user_input, TRIGGER_ECHOES)
                    except Exception:
                        pass  # Logging failure shouldn't stop input
                    
                    # Attach stable message ID before enqueue so downstream dedup works
                    _msg = _ensure_id(user_input.strip())
                    # Put in queue - THIS IS CRITICAL
                    await qe.put(_msg)
                
            except EOFError:
                # Ctrl+D pressed - normal exit
                break
            except KeyboardInterrupt:
                # Ctrl+C pressed - normal exit
                break
            except Exception as e:
                # Input error - log but CONTINUE (don't break!)
                try:
                    await bomb_log(f"Input error: {e}", BLUE_WHISPERS)
                except Exception:
                    pass
                # Continue loop - don't break!
            
            # Small yield to other tasks
            await asyncio.sleep(0.01)
            
        except Exception as e:
            # Outer exception handler - also don't break!
            await asyncio.sleep(0.1)  # Small delay before retry


async def neon_input(qe: asyncio.Queue[str]) -> None:
    """Read user input and feed it into the provided queue.

    Parameters
    ----------
    qe : asyncio.Queue[str]
        Target queue for sanitized user input.
    """
    try:
        if str(os.getenv("JINX_STARTUP_VALIDATE", "0")).strip().lower() not in ("", "0", "false", "off", "no"):
            while not jx_state.shutdown_event.is_set():
                await asyncio.sleep(0.2)
            return
    except Exception:
        pass
    try:
        force_simple = str(os.getenv("JINX_FORCE_SIMPLE_INPUT", "0")).strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        force_simple = False
    if force_simple:
        try:
            jx_state.ui_prompt_toolkit = False
        except Exception:
            pass
        return await _simple_input_loop(qe)
    # Try to use prompt_toolkit, fallback to simple input if fails
    use_prompt_toolkit = True
    PromptSession = None
    KeyBindings = None
    FormattedText = None
    
    try:
        # Lazily import prompt_toolkit symbols
        _ptk = importlib.import_module("prompt_toolkit")
        _ptk_keys = importlib.import_module("prompt_toolkit.key_binding")
        _ptk_fmt = importlib.import_module("prompt_toolkit.formatted_text")
        PromptSession = getattr(_ptk, "PromptSession")
        KeyBindings = getattr(_ptk_keys, "KeyBindings")
        FormattedText = getattr(_ptk_fmt, "FormattedText")
        
        # Test if prompt_toolkit can initialize (quick test)
        # Don't actually create session here to avoid blocking
        
    except Exception as e:
        # Prompt_toolkit не работает, используем fallback
        use_prompt_toolkit = False
        try:
            await bomb_log(f"prompt_toolkit unavailable (fallback to simple input): {e}", BLUE_WHISPERS)
        except Exception:
            pass  # Even logging can fail, don't care
    
    if not use_prompt_toolkit or PromptSession is None:
        # Simple input fallback для Windows PowerShell
        try:
            jx_state.ui_prompt_toolkit = False
        except Exception:
            pass
        return await _simple_input_loop(qe)

    # Try to create session - if this fails, use simple input
    try:
        finger_wire = KeyBindings()
    except Exception as e:
        print(f"KeyBindings failed: {e}")
        return await _simple_input_loop(qe)

    def _toolbar() -> "FormattedText":
        show_det = parse_env_bool("JINX_SPINNER_ACTIVITY_DETAIL", True)
        # Build activity line
        act = getattr(jx_state, "activity", "") or ""
        act_ts = float(getattr(jx_state, "activity_ts", 0.0) or 0.0)
        spin_t0 = float(getattr(jx_state, "spin_t0", 0.0) or 0.0)
        pulse = int(getattr(jx_state, "pulse", 0) or 0)
        now = time.perf_counter()
        age = (now - act_ts) if act_ts else 0.0
        total = (now - spin_t0) if spin_t0 else 0.0
        det = getattr(jx_state, "activity_detail", None)
        det_str = ""
        if show_det:
            try:
                det_str, _stage, _tasks = format_activity_detail(det)  # stage/tasks not needed here
            except Exception:
                det_str = ""
        text = f"❤ {pulse} | {act or 'ready'} [{age:.1f}s]{det_str} (total {total:.1f}s)"
        return FormattedText([("", text)])

    # Try to create PromptSession - this is where "No Windows console" error occurs
    try:
        sess = PromptSession(key_bindings=finger_wire, bottom_toolbar=_toolbar)
    except Exception as e:
        # PromptSession failed - use simple fallback
        try:
            jx_state.ui_prompt_toolkit = False
        except Exception:
            pass
        return await _simple_input_loop(qe)
    try:
        jx_state.ui_prompt_toolkit = True
    except Exception:
        pass
    boom_clock: dict[str, float] = {"time": asyncio.get_running_loop().time()}
    activity = asyncio.Event()
    # Gate to emit <no_response> only once per inactivity period (until next activity)
    noresp_sent: dict[str, bool] = {"v": False}

    @finger_wire.add("<any>")
    def _(triggerbit) -> None:  # prompt_toolkit callback
        boom_clock["time"] = asyncio.get_running_loop().time()
        triggerbit.app.current_buffer.insert_text(triggerbit.key_sequence[0].key)
        # Reset no-response gate and signal activity to reset the timer immediately
        noresp_sent["v"] = False
        activity.set()
        # Short-lived preemption: raise throttle briefly on user typing to keep UI responsive
        try:
            set_throttle_ttl(0.35)
        except Exception:
            # Fallback to legacy behavior if backpressure module unavailable
            try:
                jx_state.throttle_event.set()
                setattr(jx_state, "throttle_unset_ts", float(time.perf_counter()) + 0.35)
            except Exception:
                pass

    async def kaboom_watch() -> None:
        """Emit <no_response> after inactivity using a reactive timer.

        Avoids periodic polling by waiting for either activity or timeout.
        """
        while True:
            # If the agent is thinking (spinner on), pause the idle timer and gate reset
            try:
                if bool(getattr(jx_state, "spin_on", False)):
                    # Reset timer baseline while busy so the user gets full TIMEOUT after output
                    boom_clock["time"] = asyncio.get_running_loop().time()
                    noresp_sent["v"] = False
                    await asyncio.sleep(0.2)
                    continue
            except Exception:
                pass
            # Calculate remaining time based on the later of last activity or last agent reply
            now = asyncio.get_running_loop().time()
            try:
                limit = int(getattr(jx_state, "boom_limit", 30))
            except Exception:
                limit = 30
            try:
                base = max(float(boom_clock["time"]), float(getattr(jx_state, "last_agent_reply_ts", 0.0) or 0.0))
            except Exception:
                base = float(boom_clock["time"])
            remaining = max(0.0, limit - (now - base))
            activity.clear()
            try:
                # Wait for either new activity or the inactivity timeout
                await asyncio.wait_for(activity.wait(), timeout=remaining)
                # Activity occurred: loop to recalculate remaining
                continue
            except asyncio.TimeoutError:
                # Timeout: no activity within boom_limit
                if not noresp_sent["v"]:
                    await blast_mem("<no_response>")
                    await bomb_log("<no_response>", TRIGGER_ECHOES)
                    # Do not enqueue <no_response> into the main conversation pipeline.
                    # It is an internal signal only; enqueuing can cause infinite auto-replies.
                    noresp_sent["v"] = True
                    boom_clock["time"] = asyncio.get_running_loop().time()
                else:
                    # Already emitted for this inactivity period; wait until activity
                    activity.clear()
                    try:
                        await activity.wait()
                    except Exception:
                        pass

    watch_task = asyncio.create_task(kaboom_watch())
    # Periodic invalidate to refresh toolbar while spinner runs
    async def toolbar_pulse() -> None:
        try:
            while True:
                await asyncio.sleep(0.12)
                try:
                    # Refresh when spinner is on OR when there is activity/detail to show
                    spin_on = bool(getattr(jx_state, "spin_on", False))
                    act = (getattr(jx_state, "activity", "") or "").strip()
                    det = getattr(jx_state, "activity_detail", None)
                    if spin_on or act or det:
                        sess.app.invalidate()
                except Exception:
                    pass
        except asyncio.CancelledError:
            return
    pulse_task = asyncio.create_task(toolbar_pulse())
    try:
        prompt_task: asyncio.Task[str] | None = None
        shutdown_task: asyncio.Task[bool] | None = None  # Event.wait() resolves to True
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            # If too many consecutive errors, fall back to simple input
            if consecutive_errors >= max_consecutive_errors:
                print("\n[Switching to simple input mode due to errors]")
                return await _simple_input_loop(qe)
            try:
                # Race the prompt against a shutdown signal to exit promptly
                try:
                    pt: asyncio.Task[str] = asyncio.create_task(sess.prompt_async("\n"))
                except Exception as e:
                    # prompt_async failed - switch to simple input
                    watch_task.cancel()
                    pulse_task.cancel()
                    with contextlib.suppress(Exception):
                        await watch_task
                        await pulse_task
                    return await _simple_input_loop(qe)
                
                # Ensure any exception (including BaseException like KeyboardInterrupt) is retrieved
                def _swallow_task_exc(t: asyncio.Task) -> None:
                    try:
                        # Using result() to also re-raise BaseException subclasses
                        _ = t.result()
                    except KeyboardInterrupt:
                        # Don't cancel during KeyboardInterrupt - let it propagate naturally
                        return
                    except asyncio.CancelledError:
                        # Already cancelled, nothing to do
                        return
                    except BaseException:
                        pass
                pt.add_done_callback(_swallow_task_exc)
                st: asyncio.Task[bool] = asyncio.create_task(jx_state.shutdown_event.wait())
                tasks: set[asyncio.Future[Any]] = {cast(asyncio.Future[Any], pt), cast(asyncio.Future[Any], st)}
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if st in done:
                    # Politely ask prompt_toolkit to exit instead of cancelling the task
                    try:
                        sess.app.exit(exception=EOFError())
                    except Exception:
                        pass
                    # Ensure prompt_task finishes and swallow any BaseException (e.g., KeyboardInterrupt)
                    with contextlib.suppress(BaseException):
                        await pt
                    break
                # Got user input
                st.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await st
                v: str = pt.result()
                if v.strip():
                    _msg = _ensure_id(v.strip())
                    await bomb_log(_msg, TRIGGER_ECHOES)
                    await qe.put(_msg)
                    consecutive_errors = 0  # Reset on success
                # expose tasks for final cleanup
                prompt_task = pt
                shutdown_task = st
            except (EOFError, KeyboardInterrupt):
                # Treat both as a clean exit of input loop
                if prompt_task is not None:
                    with contextlib.suppress(BaseException):
                        await prompt_task
                break
            except Exception as e:  # pragma: no cover - guard rail for TTY issues
                consecutive_errors += 1
                await bomb_log(f"ERROR INPUT chaos keys went rogue: {e}")
                await asyncio.sleep(0.5)  # Brief delay before retry
    finally:
        # Ensure watchdog is cancelled when input loop exits
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
        pulse_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pulse_task
        # Force-close prompt_toolkit app and consume any leftover tasks
        with contextlib.suppress(Exception):
            sess.app.exit(exception=EOFError())
        if prompt_task is not None:
            pt2 = cast(asyncio.Task[Any], prompt_task)
            if not pt2.done():
                with contextlib.suppress(BaseException):
                    await pt2
        if shutdown_task is not None:
            st2 = cast(asyncio.Task[Any], shutdown_task)
            if not st2.done():
                st2.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await st2
