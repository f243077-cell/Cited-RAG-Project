"""The three read-only tools over the existing retriever, docstore and manifest.

Callers hold the index lock when there is one: the API runs a tool under `index_lock`, the agent through the pipeline's
`_indexed`, so this module never locks and can never deadlock with them. A tool problem comes back as a ToolResult with
ok=false (F-TOOL-6); only a bug raises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from citerag.config import Settings
from citerag.errors import ValidationFailedError
from citerag.ingest.manifest import Manifest, document_infos
from citerag.rag.context import Chunk
from citerag.tools.contracts import (
    ARGS,
    ReadPageArgs,
    SearchArgs,
    ToolHit,
    ToolName,
    ToolResult,
)

NO_RELEVANT = "no relevant passages"


class ToolRetriever(Protocol):
    def candidates(self, query: str, doc_ids: list[str] | None = None) -> list[Chunk]: ...

    def embed_query(self, query: str) -> list[float]: ...

    def top_cosine(self, query_embedding: list[float], doc_ids: list[str] | None = None) -> float: ...

    def page_chunks(self, doc_id: str, page: int) -> list[Chunk]: ...


def validate_args(name: ToolName, args: dict[str, Any]) -> BaseModel:
    """Raises validation_error with the reason, never echoing the input."""
    try:
        return ARGS[name].model_validate(args)
    except ValidationError as exc:
        reasons = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())
        raise ValidationFailedError(f"{name}: {reasons}") from exc


def _hit(chunk: Chunk) -> ToolHit:
    kind = "image" if chunk.kind == "image" else "text"
    return ToolHit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_name=chunk.doc_name,
        page=chunk.page,
        kind=kind,
        text=chunk.text,
        score=chunk.score or None,
        image_path=chunk.image_path,
    )


def cap_hits(hits: list[ToolHit], max_chars: int) -> tuple[list[ToolHit], bool]:
    """Keep whole hits in order until the budget runs out; cut the first that does not fit, drop the rest."""
    kept, used = [], 0
    for hit in hits:
        room = max_chars - used
        if room <= 0:
            return kept, True
        if len(hit.text) > room:
            kept.append(hit.model_copy(update={"text": hit.text[:room]}))
            return kept, True
        kept.append(hit)
        used += len(hit.text)
    return kept, False


class ToolLibrary:
    def __init__(
        self,
        settings: Settings,
        retriever: ToolRetriever,
        manifest: Callable[[], Manifest | None],
        scope: list[str] | None = None,
    ) -> None:
        """`scope` is the request's document filter (F-TOOL-7): no tool call can reach past it."""
        self._cfg, self._retriever, self._manifest, self._scope = settings, retriever, manifest, scope

    def run(self, name: ToolName, args: dict[str, Any]) -> ToolResult:
        try:
            checked = validate_args(name, args)
        except ValidationFailedError as exc:
            return ToolResult(tool=name, ok=False, error=exc.message)
        if name == "list_documents":
            return self.list_documents()
        if name == "search_documents":
            return self.search_documents(checked)
        return self.read_page(checked)

    def _known(self) -> Manifest | None:
        return self._manifest()

    def _allowed(self, manifest: Manifest) -> set[str]:
        ids = set(manifest.docs)
        return ids if self._scope is None else ids & set(self._scope)

    def list_documents(self) -> ToolResult:
        manifest = self._known()
        if manifest is None:
            return ToolResult(tool="list_documents", ok=True, message="no documents yet")
        allowed = self._allowed(manifest)
        documents = [info for info in document_infos(manifest) if info.doc_id in allowed]
        return ToolResult(tool="list_documents", ok=True, documents=documents)

    def search_documents(self, args: SearchArgs) -> ToolResult:
        manifest = self._known()
        if manifest is None or not manifest.docs:
            return ToolResult(tool="search_documents", ok=True, message="no documents yet")
        allowed = self._allowed(manifest)
        if args.doc_ids is not None:
            unknown = sorted(set(args.doc_ids) - set(manifest.docs))
            if unknown:
                return ToolResult(
                    tool="search_documents", ok=False, error=f"unknown doc_ids: {', '.join(unknown)}"
                )
            allowed &= set(args.doc_ids)
        doc_ids = None if allowed == set(manifest.docs) else sorted(allowed)
        if doc_ids == []:
            return ToolResult(tool="search_documents", ok=True, message="no documents in scope")
        top = self._retriever.top_cosine(self._retriever.embed_query(args.query), doc_ids)
        if top < self._cfg.min_top_similarity:  # F-TOOL-4: the gate, with no rewrite request
            return ToolResult(tool="search_documents", ok=True, message=NO_RELEVANT, top_similarity=top)
        found = [_hit(chunk) for chunk in self._retriever.candidates(args.query, doc_ids)[: args.k]]
        hits, truncated = cap_hits(found, self._cfg.agent_tool_result_chars)
        return ToolResult(
            tool="search_documents", ok=True, hits=hits, top_similarity=top, truncated=truncated
        )

    def read_page(self, args: ReadPageArgs) -> ToolResult:
        manifest = self._known()
        if manifest is None:
            return ToolResult(tool="read_page", ok=False, error="no documents yet")
        allowed = self._allowed(manifest)
        doc_id = args.doc if args.doc in manifest.docs else None
        if doc_id is None:  # a name resolves only to a stored name, never to a path
            doc_id = next((i for i, entry in manifest.docs.items() if entry.name == args.doc), None)
        if doc_id is None or doc_id not in allowed:
            return ToolResult(tool="read_page", ok=False, error=f"unknown document: {args.doc}")
        entry = manifest.docs[doc_id]
        if entry.pages is not None and args.page > entry.pages:
            return ToolResult(tool="read_page", ok=False, error=f"{entry.name} has {entry.pages} pages")
        found = [_hit(c) for c in self._retriever.page_chunks(doc_id, args.page)]
        if not found:
            return ToolResult(tool="read_page", ok=True, message=f"no text stored for page {args.page}")
        hits, truncated = cap_hits(found, self._cfg.agent_tool_result_chars)
        return ToolResult(tool="read_page", ok=True, hits=hits, truncated=truncated)
