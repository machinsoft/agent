from __future__ import annotations

import shutil
import textwrap
import asyncio
import importlib
import sys
from .locks import get_print_lock


def pretty_echo(text: str, title: str = "Jinx") -> None:
    """Render model output in a neat ASCII box with a title.

    - Uses word-wrapping (no mid-word splits) for readability.
    - Preserves blank lines from the original text.
    - Avoids ANSI so it won't clash with prompt rendering.
    """
    try:
        import jinx.state as _jx_state
        _jx_state.ui_output_active = True
    except Exception:
        _jx_state = None  # type: ignore
    try:
        # Clear any in-place spinner line before printing the box
        try:
            sys.stdout.write("\r" + (" " * 200) + "\r")
            sys.stdout.flush()
        except Exception:
            pass

        width = shutil.get_terminal_size((80, 24)).columns
        width = max(50, min(width, 120))
        inner_w = width - 2

        # Title bar
        title_str = f" {title} " if title else ""
        title_len = len(title_str)
        if title_len and title_len + 2 < inner_w:
            top = "+-" + title_str + ("-" * (inner_w - title_len - 2)) + "+"
        else:
            top = "+" + ("-" * inner_w) + "+"
        bot = "+" + ("-" * inner_w) + "+"

        print(top)
        lines = text.splitlines() if text else [""]
        for ln in lines:
            wrapped = (
                textwrap.wrap(
                    ln,
                    width=inner_w,
                    break_long_words=False,
                    break_on_hyphens=False,
                    replace_whitespace=False,
                )
                if ln.strip() != ""
                else [""]
            )
            for chunk in wrapped:
                pad = inner_w - len(chunk)
                print(f"|{chunk}{' ' * pad}|")
        print(bot + "\n")
    finally:
        try:
            if _jx_state is not None:
                _jx_state.ui_output_active = False
        except Exception:
            pass


async def pretty_echo_async(text: str, title: str = "Jinx") -> None:
    """Async variant of pretty_echo with cooperative yields and PTK-safe stdout.

    - Uses prompt_toolkit.patch_stdout to avoid TTY contention with the active prompt.
    - Yields to the event loop every few lines to keep input responsive.
    """
    try:
        patch_stdout = importlib.import_module("prompt_toolkit.patch_stdout").patch_stdout  # type: ignore[assignment]
        print_formatted_text = importlib.import_module("prompt_toolkit").print_formatted_text  # type: ignore[assignment]
        FormattedText = importlib.import_module("prompt_toolkit.formatted_text").FormattedText  # type: ignore[assignment]
    except Exception:
        # Fallback to sync printing in a thread if PTK unavailable
        await asyncio.to_thread(pretty_echo, text, title)
        return

    # Windows consoles can be unreliable with prompt_toolkit stdout patching,
    # but if prompt_toolkit PromptSession is active, patch_stdout is the best way
    # to avoid corrupting the input line.
    try:
        if sys.platform.startswith("win"):
            try:
                import jinx.state as _jx_state
                use_ptk = bool(getattr(_jx_state, "ui_prompt_toolkit", False))
            except Exception:
                use_ptk = False
            if not use_ptk:
                await asyncio.to_thread(pretty_echo, text, title)
                return
    except Exception:
        await asyncio.to_thread(pretty_echo, text, title)
        return

    # Use micro-modular print lock
    async with get_print_lock():
        try:
            import jinx.state as _jx_state
            _jx_state.ui_output_active = True
        except Exception:
            _jx_state = None  # type: ignore
        async def _ptk_print() -> None:
            width = shutil.get_terminal_size((80, 24)).columns
            width = max(50, min(width, 120))
            inner_w = width - 2

            title_str = f" {title} " if title else ""
            title_len = len(title_str)
            if title_len and title_len + 2 < inner_w:
                top = "+-" + title_str + ("-" * (inner_w - title_len - 2)) + "+"
            else:
                top = "+" + ("-" * inner_w) + "+"
            bot = "+" + ("-" * inner_w) + "+"

            ft = FormattedText
            with patch_stdout(raw=True):
                print_formatted_text(ft([("", top)]))
                if not text:
                    print_formatted_text(ft([("", f"|{' ' * inner_w}|")]))
                else:
                    count = 0
                    for ln in text.splitlines():
                        wrapped = (
                            textwrap.wrap(
                                ln,
                                width=inner_w,
                                break_long_words=False,
                                break_on_hyphens=False,
                                replace_whitespace=False,
                            )
                            if ln.strip() != ""
                            else [""]
                        )
                        for chunk in wrapped:
                            pad = inner_w - len(chunk)
                            print_formatted_text(ft([("", f"|{chunk}{' ' * pad}|")]))
                            count += 1
                            if (count % 20) == 0:
                                await asyncio.sleep(0)
                print_formatted_text(ft([("", bot + "\n")]))
        try:
            await asyncio.wait_for(_ptk_print(), timeout=0.6)
        except Exception:
            await asyncio.to_thread(pretty_echo, text, title)
        finally:
            try:
                if _jx_state is not None:
                    _jx_state.ui_output_active = False
            except Exception:
                pass
