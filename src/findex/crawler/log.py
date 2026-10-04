"""Per-URL crawl log. One line per attempt, flushed so a crash keeps the record."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from findex.crawler.fetch import Attempt


def format_log_line(
    *,
    ts: str,
    url: str,
    status: int | str | None,
    nbytes: int,
    elapsed: float,
    attempt: int,
    error: str | None,
    retry_after: float | None = None,
) -> str:
    status_text = "-" if status is None else str(status)
    error_text = error if error else "-"
    line = (
        f"ts={ts} url={url} status={status_text} bytes={nbytes} "
        f"elapsed={elapsed:.3f} attempt={attempt} error={error_text}"
    )
    if retry_after is not None:
        line += f" retry_after={retry_after:.3f}"
    return line


class CrawlLog:
    """Append-only log opened once per crawl and closed with the client."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def attempt(self, url: str, number: int, record: Attempt) -> None:
        self.line(
            url=url,
            status=record.status,
            nbytes=record.nbytes,
            elapsed=record.elapsed,
            attempt=number,
            error=record.error,
            retry_after=record.retry_after,
            ts=record.at,
        )

    def note(self, url: str, status: str, error: str) -> None:
        self.line(
            url=url,
            status=status,
            nbytes=0,
            elapsed=0.0,
            attempt=1,
            error=error,
            retry_after=None,
        )

    def line(
        self,
        *,
        url: str,
        status: int | str | None,
        nbytes: int,
        elapsed: float,
        attempt: int,
        error: str | None,
        retry_after: float | None,
        ts: str | None = None,
    ) -> None:
        if ts is None:
            ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        text = format_log_line(
            ts=ts,
            url=url,
            status=status,
            nbytes=nbytes,
            elapsed=elapsed,
            attempt=attempt,
            error=error,
            retry_after=retry_after,
        )
        self._fh.write(text + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
