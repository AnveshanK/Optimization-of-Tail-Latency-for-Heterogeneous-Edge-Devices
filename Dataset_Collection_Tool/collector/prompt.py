"""Questions that never block an unattended collection (standard library only: also used by run.py).

timed_input() waits for one line of keyboard input for at most `timeout` seconds. It does not leave a
background reader behind (select() on Linux / macOS, console polling on Windows), so a later question
in another process (the collector) still receives what is typed.
"""

import os
import sys
import time


def interactive():
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def timed_input(prompt, timeout):
    """The typed line (stripped; '' = just Enter), None after `timeout` seconds without an answer.
    Raises EOFError when no keyboard is attached (input closed or piped and exhausted)."""
    print(prompt, end="", flush=True)
    try:
        return _read_line(timeout)
    except EOFError:
        print()
        raise


def _read_line(timeout):
    if sys.stdin is None:
        raise EOFError
    if os.name == "nt":
        if not interactive():
            return input().strip()               # piped input; EOFError at its end
        import msvcrt
        buf, end = "", time.time() + timeout
        while time.time() < end:
            if not msvcrt.kbhit():
                time.sleep(0.05)
                continue
            ch = msvcrt.getwche()
            if ch in "\r\n":
                print()
                return buf.strip()
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\b":
                buf = buf[:-1]
                print(" \b", end="", flush=True)
            else:
                buf += ch
        print()
        return None
    import select
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        return input().strip()
    if not ready:
        print()
        return None
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.strip()


def ask(question, default, choices=None, timeout=600.0, say=print):
    """Ask with a default that is taken on Enter, after `timeout` seconds without an answer, or when
    no keyboard is attached - an unattended start never hangs on a question."""
    hint = f" [{default}]" if default else ""
    while True:
        try:
            ans = timed_input(f"{question}{hint}: ", timeout)
        except EOFError:
            if default:
                say(f"  (no keyboard input - using the default: {default})")
                return default
            raise SystemExit("an answer is needed; pass it on the command line (see --help)")
        if ans is None:
            if default:
                say(f"  (no answer within {timeout / 60:.0f} min - using the default: {default})")
                return default
            continue
        ans = ans or (default or "")
        if not ans:
            continue
        if choices and ans.lower() not in choices:
            say(f"  please answer one of: {', '.join(choices)}")
            continue
        return ans.lower() if choices else ans
