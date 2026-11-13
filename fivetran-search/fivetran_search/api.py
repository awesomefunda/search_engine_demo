"""HTTP layer. Thin by design: it validates input and delegates to SearchService.

    uvicorn fivetran_search.api:app --port 8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import SETTINGS
from .pipeline import SearchService

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="fivetran.com internal search", version="0.1.0")
service = SearchService()


@app.get("/api/search")
def api_search(
    q: str = Query("", description="keyword or full sentence"),
    cursor: str | None = Query(None, description="opaque cursor from a previous response"),
    page_size: int = Query(SETTINGS.page_size, ge=1, le=50),
) -> JSONResponse:
    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    resp = service.search(q, cursor_token=cursor, page_size=page_size)
    return JSONResponse(resp.as_dict())


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(service.status())


@app.post("/api/refresh")
def api_refresh(force: bool = Query(True)) -> JSONResponse:
    """Recrawl. Conditional GETs mean an unchanged site costs almost nothing."""
    return JSONResponse(service.refresh(force=force))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
