#!/usr/bin/env python3
"""Stream a command and terminate its process group after prolonged silence."""

from __future__ import annotations

import argparse
import math
import os
import selectors
import signal
import subprocess
import sys
import time


FATAL_OUTPUT = (
    b"OIDN Error:",
    b"Error: unsupported device type: CUDA",
    b"Error: invalid handle",
    b"device has unsupported compute capability",
    b"OIDN CUDA device error",
)


def _terminate_group(process: subprocess.Popen[bytes], grace_seconds: float = 10.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The shell leader can exit before a hung simulator grandchild.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be non-negative")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    last_output = time.monotonic()
    timed_out = False
    fatal_output = None
    output_tail = b""
    tail_length = max(map(len, FATAL_OUTPUT)) - 1

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    try:
        while True:
            ready = selector.select(timeout=1.0)
            if ready:
                chunk = os.read(process.stdout.fileno(), 65536)
                if chunk:
                    os.write(sys.stdout.fileno(), chunk)
                    last_output = time.monotonic()
                    searchable = output_tail + chunk
                    fatal_output = next(
                        (pattern.decode() for pattern in FATAL_OUTPUT if pattern in searchable),
                        None,
                    )
                    output_tail = searchable[-tail_length:]
                    if fatal_output is not None:
                        print(
                            f"\n[MiniWAM][ERROR] Fatal renderer output detected "
                            f"({fatal_output!r}); terminating the evaluator. "
                            "This job is failed and the current partial episode must not be counted.",
                            file=sys.stderr,
                            flush=True,
                        )
                        _terminate_group(process)
                        break
                elif process.poll() is not None:
                    break
            elif process.poll() is not None:
                break

            if (
                args.timeout_seconds > 0
                and time.monotonic() - last_output >= args.timeout_seconds
            ):
                timed_out = True
                print(
                    f"\n[MiniWAM][ERROR] Evaluator produced no output for "
                    f"{args.timeout_seconds:g}s; terminating the wedged process group. "
                    "This job is failed and its partial episodes must not be counted.",
                    file=sys.stderr,
                    flush=True,
                )
                _terminate_group(process)
                break
    except BaseException:
        _terminate_group(process)
        raise
    finally:
        selector.close()

    if timed_out:
        return 124
    if fatal_output is not None:
        return 125
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
