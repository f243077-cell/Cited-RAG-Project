"""POST /v1/tools/{name}: the read-only tools over HTTP, so `citerag mcp` never opens Chroma itself (ADR-21)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends

from citerag.api.deps import Services, get_services, require_api_key
from citerag.tools.contracts import ToolName, ToolResult
from citerag.tools.library import ToolLibrary, validate_args

router = APIRouter(prefix="/v1/tools", dependencies=[Depends(require_api_key)])


@router.post("/{name}")
def run_tool(
    name: ToolName,
    services: Annotated[Services, Depends(get_services)],
    args: Annotated[dict[str, Any] | None, Body()] = None,
) -> ToolResult:
    """Bad arguments are a 400 here; a tool-level problem such as an unknown document is ok=false."""
    validate_args(name, args or {})
    library = ToolLibrary(services.deps.cfg, services.retriever, lambda: services.ingestor.manifest)
    with services.deps.index_lock:  # an upload rebuilds BM25; a search must not see half of it
        return library.run(name, args or {})
