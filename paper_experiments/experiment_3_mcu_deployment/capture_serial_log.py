"""Capture a Pico USB CDC text log until a marker or timeout is reached."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import serial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--open-timeout", type=float, default=30.0)
    parser.add_argument("--stop-marker", default="timing runs=")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_deadline = time.monotonic() + args.open_timeout
    connection = None
    while connection is None:
        try:
            connection = serial.Serial(args.port, args.baud, timeout=0.25)
        except serial.SerialException:
            if time.monotonic() >= open_deadline:
                raise
            time.sleep(0.1)

    deadline = time.monotonic() + args.timeout
    captured: list[str] = []
    with connection, args.output.open("w", encoding="utf-8", newline="\n") as handle:
        while time.monotonic() < deadline:
            raw = connection.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            captured.append(line)
            print(line, flush=True)
            handle.write(line + "\n")
            handle.flush()
            if args.stop_marker and args.stop_marker in line:
                break

    if not captured:
        raise TimeoutError(f"No serial output captured from {args.port}")


if __name__ == "__main__":
    main()
