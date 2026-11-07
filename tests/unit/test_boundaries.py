"""Framework boundaries (CLAUDE.md rule 1, ADR-15 to ADR-17). LangGraph and the MCP SDK were importable before
Phase 3 as transitive dependencies, so only a test keeps them in their packages."""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "citerag"
RULES = {
    "langgraph": "agent_graph",
    "mcp": "mcp_server",
    "crewai": "agents",
}
NO_FRAMEWORK = (
    "llama_index",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "crewai",
    "mcp",
    "chromadb",
)


def imports(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_][\w]*)", text, re.M))


@pytest.mark.parametrize(("module", "home"), RULES.items())
def test_a_framework_is_imported_only_in_its_package(module, home):
    offenders = [p for p in SRC.rglob("*.py") if module in imports(p) and home not in p.parts]
    assert offenders == []


def test_tools_import_no_framework():
    for path in (SRC / "tools").rglob("*.py"):
        assert not imports(path) & set(NO_FRAMEWORK), path
