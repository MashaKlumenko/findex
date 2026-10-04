"""HTTP API and the search page. ``uvicorn findex.web:app`` loads ``app``."""

from findex.web.app import app, create_app

__all__ = ["app", "create_app"]
