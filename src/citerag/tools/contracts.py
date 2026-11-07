"""Tool arguments and results (PHASE3_ARCHITECTURE section 5). Public through MCP and POST /v1/tools/{name}."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from citerag.schemas import DocumentInfo

ToolName = Literal["list_documents", "search_documents", "read_page"]
TOOL_NAMES: tuple[ToolName, ...] = ("list_documents", "search_documents", "read_page")
MAX_K = 10
MAX_QUERY_CHARS = 500


class ListDocumentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS, description="What to search for")
    doc_ids: list[str] | None = Field(default=None, max_length=20, description="Only these documents")
    k: int = Field(default=5, ge=1, le=MAX_K, description="How many passages to return")


class ReadPageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc: str = Field(min_length=1, max_length=200, description="A doc_id or the exact document name")
    page: int = Field(ge=1, le=10_000, description="Page number, counting from 1")


ARGS: dict[ToolName, type[BaseModel]] = {
    "list_documents": ListDocumentsArgs,
    "search_documents": SearchArgs,
    "read_page": ReadPageArgs,
}


class ToolHit(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int | None = None
    kind: Literal["text", "image"] = "text"
    text: str  # untrusted document text: reaches a model only through render_passages
    score: float | None = None
    image_path: str | None = None


class ToolResult(BaseModel):
    tool: ToolName
    ok: bool
    error: str | None = None  # a short typed message, never a stack trace
    message: str = ""  # written by code, e.g. "no relevant passages"
    hits: list[ToolHit] = Field(default_factory=list)
    documents: list[DocumentInfo] = Field(default_factory=list)
    top_similarity: float | None = None  # search only: the relevance-gate value
    truncated: bool = False


DESCRIPTIONS: dict[ToolName, str] = {
    "list_documents": "List the documents in the library.",
    "search_documents": "Search the documents. Returns passages, or none when nothing is relevant.",
    "read_page": "Read the stored text of one page of one document.",
}


ACT_TOOLS: tuple[ToolName, ...] = (
    "search_documents",
    "read_page",
)  # list_documents distracted the model in the M10 spike


def tool_specs(names: tuple[ToolName, ...] = TOOL_NAMES) -> list[dict]:
    """Plain JSON schemas for the model, so no framework type crosses into llm/."""
    return [
        {"name": name, "description": DESCRIPTIONS[name], "parameters": ARGS[name].model_json_schema()}
        for name in names
    ]
