"""Dependencies. Tests replace ``get_index`` via ``dependency_overrides``."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from findex.index import Index
from findex.web.settings import Settings, load_settings


def get_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return load_settings()


def get_index(request: Request) -> Index | None:
    """The process-wide index, or ``None`` when it has not been loaded."""
    index = getattr(request.app.state, "index", None)
    if isinstance(index, Index):
        return index
    return None


def require_index(index: Annotated[Index | None, Depends(get_index)]) -> Index:
    if index is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    return index
