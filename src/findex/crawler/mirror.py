"""Local HTML mirror for the lab benchmark and the live demo.

The process is ours: ``robots.txt`` allows every user agent, and nothing here
contacts another machine. ``--latency`` sleeps inside the server thread so the
client spends its time waiting, which is the workload asyncio is for.
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MirrorServer(ThreadingHTTPServer):
    """Tiny site: ``/page/N``, one disallowed path, one 429, one redirect loop."""

    def __init__(
        self,
        latency: float = 0.0,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        super().__init__((host, port), Handler)
        self.latency = latency
        self.flaky_hits = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802  (stdlib handler name)
        server = self.server
        if not isinstance(server, MirrorServer):
            self._send(500, b"bad server", "text/plain")
            return
        if server.latency > 0:
            time.sleep(server.latency)
        path = self.path.split("?", 1)[0]
        if path == "/robots.txt":
            body = b"User-agent: *\nAllow: /\nDisallow: /private\n"
            self._send(200, body, "text/plain; charset=utf-8")
            return
        if path == "/private":
            body = _page("Private", "this path is disallowed", [])
            self._send(200, body, "text/html")
            return
        if path == "/flaky":
            server.flaky_hits += 1
            if server.flaky_hits == 1:
                body = b"slow down"
                self.send_response(429)
                self.send_header("Retry-After", "1")
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send(
                200,
                _page("Recovered", "ok after 429", []),
                "text/html; charset=utf-8",
            )
            return
        if path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path.startswith("/page/"):
            try:
                number = int(path.removeprefix("/page/"))
            except ValueError:
                self._send(404, b"missing", "text/plain")
                return
            if number == 0:
                links = [f"/page/{i}" for i in range(1, 251)]
            else:
                links = [f"/page/{number + offset}" for offset in range(1, 8)]
            text = (
                f"Document {number} about cooperative scheduling of coroutines "
                f"in an asyncio crawler."
            )
            body = _page(f"Page {number}", text, links)
            self._send(200, body, "text/html; charset=utf-8")
            return
        self._send(404, b"missing", "text/plain")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def _page(title: str, text: str, links: list[str]) -> bytes:
    anchors = "".join(f'<a href="{href}"></a>' for href in links)
    html = (
        f"<html><head><title>{title}</title></head>"
        f"<body><p>{text}</p>{anchors}</body></html>"
    )
    return html.encode("utf-8")


def serve_in_thread(
    latency: float = 0.0,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[MirrorServer, threading.Thread]:
    server = MirrorServer(latency=latency, host=host, port=port)
    thread = threading.Thread(
        target=server.serve_forever, name="findex-mirror", daemon=True
    )
    thread.start()
    return server, thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Local mirror for findex crawl.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--latency",
        type=float,
        default=0.15,
        help="Seconds the server waits before each response.",
    )
    args = parser.parse_args()
    server, _thread = serve_in_thread(
        latency=args.latency, host=args.host, port=args.port
    )
    bound = server.server_address
    print(f"mirror http://{bound[0]}:{bound[1]}/page/0  latency={args.latency}s")
    print("robots.txt allows all agents except Disallow: /private")
    try:
        thread_join = threading.Event()
        thread_join.wait()
    except KeyboardInterrupt:
        print("\nstopping")
        server.shutdown()


if __name__ == "__main__":
    main()
