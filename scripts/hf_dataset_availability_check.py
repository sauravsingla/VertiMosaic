#!/usr/bin/env python3
"""Rate-limited availability checks for the VertiMosaic Hugging Face dataset API."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ENDPOINT = (
    "https://huggingface.co/api/datasets/"
    "sauravsingla08/VertiMosaic-VFL-Benchmark"
)


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with conservative public-service limits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=900)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--output", default="hf_availability_900.csv")
    parser.add_argument("--summary", default="hf_availability_summary.json")
    parser.add_argument("--max-consecutive-errors", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Enforce a bounded, rate-limited metadata-only test."""
    if not 1 <= args.requests <= 900:
        raise SystemExit("--requests must be between 1 and 900")
    if args.delay < 3.0:
        raise SystemExit("--delay must be at least 3.0 seconds")
    if args.jitter < 0:
        raise SystemExit("--jitter must be non-negative")
    if args.endpoint != DEFAULT_ENDPOINT:
        raise SystemExit("Only the VertiMosaic Hugging Face metadata endpoint is allowed")


def main() -> None:
    """Run the availability check and write CSV plus JSON evidence."""
    args = parse_args()
    validate_args(args)

    output = Path(args.output)
    summary_path = Path(args.summary)
    successes = 0
    failures = 0
    consecutive_errors = 0
    latencies: list[float] = []
    started_at = utc_now()

    headers = {
        "Accept": "application/json",
        "Cache-Control": "no-cache",
        "User-Agent": (
            "VertiMosaic-availability-check/1.0 "
            "(rate-limited metadata availability test)"
        ),
    }

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "request_no",
                "timestamp_utc",
                "status",
                "latency_ms",
                "content_length",
                "error",
            ]
        )

        for request_no in range(1, args.requests + 1):
            request_started = time.perf_counter()
            status: int | None = None
            content_length: int | None = None
            error = ""

            try:
                request = urllib.request.Request(
                    args.endpoint,
                    headers=headers,
                    method="GET",
                )
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    status = response.status
                    body = response.read()
                    content_length = len(body)
                    json.loads(body.decode("utf-8"))

                latency_ms = round((time.perf_counter() - request_started) * 1000, 2)
                latencies.append(latency_ms)
                if 200 <= status < 300:
                    successes += 1
                    consecutive_errors = 0
                else:
                    failures += 1
                    consecutive_errors += 1

            except urllib.error.HTTPError as exc:
                latency_ms = round((time.perf_counter() - request_started) * 1000, 2)
                status = exc.code
                error = f"HTTPError: {exc.reason}"
                failures += 1
                consecutive_errors += 1

                writer.writerow(
                    [request_no, utc_now(), status, latency_ms, content_length, error]
                )
                handle.flush()

                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        wait_seconds = max(float(retry_after), 60.0)
                    except (TypeError, ValueError):
                        wait_seconds = 60.0
                    print(
                        f"[{request_no}/{args.requests}] HTTP 429; "
                        f"backing off for {wait_seconds:.0f}s"
                    )
                    time.sleep(wait_seconds)
                    continue

            except Exception as exc:  # noqa: BLE001 - evidence should capture runtime failures
                latency_ms = round((time.perf_counter() - request_started) * 1000, 2)
                error = f"{type(exc).__name__}: {exc}"
                failures += 1
                consecutive_errors += 1

            writer.writerow(
                [request_no, utc_now(), status, latency_ms, content_length, error]
            )
            handle.flush()

            print(
                f"[{request_no}/{args.requests}] status={status} "
                f"latency={latency_ms:.2f}ms "
                f"{'OK' if not error and status and 200 <= status < 300 else error}"
            )

            if consecutive_errors >= args.max_consecutive_errors:
                print("Stopping after consecutive-error safety threshold")
                break

            if request_no < args.requests:
                sleep_seconds = max(
                    3.0,
                    args.delay + random.uniform(-args.jitter, args.jitter),
                )
                time.sleep(sleep_seconds)

    summary = {
        "endpoint": args.endpoint,
        "requested_checks": args.requests,
        "successful_checks": successes,
        "failed_checks": failures,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "average_latency_ms": (
            round(sum(latencies) / len(latencies), 2) if latencies else None
        ),
        "minimum_delay_seconds": 3.0,
        "purpose": "rate-limited availability testing of dataset metadata",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
