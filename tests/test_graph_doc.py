"""`scripts/graph_doc.py`: the graph's generated parts and its checks, over a tiny repo."""

from __future__ import annotations

import textwrap
from pathlib import Path

import graph_doc as g
import pytest

PACKAGE = "pkg"

SOURCES: dict[str, str] = {
    "__init__.py": "",
    "config.py": """
        LIMIT = 3
        NAME: str = "x"
    """,
    "errors.py": """
        class Oops(Exception):
            pass
    """,
    "core.py": """
        from __future__ import annotations

        import json
        from typing import TYPE_CHECKING

        import numpy as np
        from pydantic import BaseModel, Field

        from pkg import errors
        from pkg.config import LIMIT
        from pkg.config import NAME

        if TYPE_CHECKING:
            from pkg.sub.deep import Thing


        class Llm:
            WIDTH = 2

            def send(self) -> None:
                from pkg.sub import deep

                del deep


        def helper() -> int:
            return LIMIT
    """,
    "sub/__init__.py": """
        from .deep import Thing
    """,
    "sub/deep.py": """
        from ..config import LIMIT
        from . import leaf


        class Thing:
            pass
    """,
    "sub/leaf.py": """
        from pkg.errors import Oops as Renamed
    """,
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for name, body in SOURCES.items():
        path = tmp_path / "src" / PACKAGE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body).lstrip())
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_core.py").write_text("def test_it() -> None:\n    pass\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("notes\n")
    return tmp_path


@pytest.fixture
def modules(repo: Path) -> dict[str, Path]:
    return g.modules_under(repo / "src" / PACKAGE, "")


@pytest.fixture
def resolver(repo: Path, modules: dict[str, Path]) -> g.Resolver:
    return g.Resolver(repo, {**modules, **g.modules_under(repo / "tests", "tests")})


def test_modules_are_named_by_their_dotted_path(modules: dict[str, Path]) -> None:
    assert sorted(modules) == ["config", "core", "errors", "sub", "sub.deep", "sub.leaf"]
    assert modules["sub"].name == "__init__.py"


@pytest.mark.parametrize(
    ("symbol", "line"),
    [
        pytest.param("LIMIT", 1, id="an assignment"),
        pytest.param("NAME", 2, id="an annotated assignment"),
    ],
)
def test_a_modules_symbols_carry_their_line(repo: Path, symbol: str, line: int) -> None:
    found = g.symbols_in(repo / "src" / PACKAGE / "config.py")
    assert found[symbol] == line, found


@pytest.mark.parametrize(
    ("node_id", "at"),
    [
        pytest.param("core", "src/pkg/core.py", id="a module is its file"),
        pytest.param("sub", "src/pkg/sub/__init__.py", id="a package is its __init__"),
        pytest.param("core.helper", "src/pkg/core.py:26", id="a function"),
        pytest.param("core.Llm", "src/pkg/core.py:17", id="a class"),
        pytest.param("core.Llm.send", "src/pkg/core.py:20", id="a method"),
        pytest.param("core.Llm.WIDTH", "src/pkg/core.py:18", id="a class attribute"),
        pytest.param("sub.Thing", "src/pkg/sub/__init__.py:1", id="a re-export"),
        pytest.param("tests.test_core.test_it", "tests/test_core.py:1", id="a test"),
        pytest.param("docs/notes.md", "docs/notes.md", id="a path that exists"),
        pytest.param("docs/", "docs/", id="a directory"),
        pytest.param("docs/<name>.md", "docs/<name>.md", id="a pattern something matches"),
        pytest.param("core.missing", None, id="a symbol that is gone"),
        pytest.param("docs/gone.md", None, id="a path that is gone"),
        pytest.param("roadmap", None, id="a name that is not code"),
        pytest.param("core.{helper, Llm}", None, id="several symbols at once"),
    ],
)
def test_a_node_id_resolves_to_where_it_lives(
    resolver: g.Resolver, node_id: str, at: str | None
) -> None:
    assert resolver.at(node_id) == at


def _edge(edges: list[g.Edge], subject: str, target: str, how: str = "") -> g.Edge:
    found = [e for e in edges if (e.subject, e.target, e.how) == (subject, target, how)]
    assert len(found) == 1, [e for e in edges if e.subject == subject]
    return found[0]


@pytest.mark.parametrize(
    ("subject", "target", "how", "names"),
    [
        pytest.param("core", "config", "", ("LIMIT", "NAME"), id="two statements merge"),
        pytest.param("core", "errors", "", (), id="a module imported from the package"),
        pytest.param("core", "numpy", "", (), id="a third-party import"),
        pytest.param("core", "pydantic", "", ("BaseModel", "Field"), id="third-party names"),
        pytest.param("core", "sub.deep", "types only", ("Thing",), id="under TYPE_CHECKING"),
        pytest.param("core", "sub.deep", "inside a function", (), id="inside a function"),
        pytest.param("sub", "sub.deep", "", ("Thing",), id="relative from a package"),
        pytest.param("sub.deep", "config", "", ("LIMIT",), id="relative two levels up"),
        pytest.param("sub.deep", "sub.leaf", "", (), id="a sibling module"),
        pytest.param("sub.leaf", "errors", "", ("Oops",), id="an alias keeps the real name"),
    ],
)
def test_imports_are_read_from_the_source(
    modules: dict[str, Path], subject: str, target: str, how: str, names: tuple[str, ...]
) -> None:
    edges = g.imports_of(modules, PACKAGE)
    assert _edge(edges, subject, target, how).names == names


def test_the_standard_library_is_left_out(modules: dict[str, Path]) -> None:
    targets = {e.target for e in g.imports_of(modules, PACKAGE)}
    assert not targets & {"json", "typing", "__future__"}, targets


def test_import_lines_name_both_ends_and_how() -> None:
    edges = [
        g.Edge("core", "config", ("LIMIT",), ""),
        g.Edge("core", "config", ("A", "B"), "inside a function"),
        g.Edge("core", "errors", (), ""),
    ]
    assert g.render_imports(edges).split("\n") == [
        "```",
        "core —imports→ config.LIMIT",
        "core —imports→ config.{A, B}   (inside a function)",
        "core —imports→ errors",
        "```",
    ]


EDGES = [
    g.Edge("tools", "tools_config", (), ""),
    g.Edge("tools", "tools.impl", (), ""),
    g.Edge("tools", "numpy", (), ""),
    g.Edge("evals.runner", "llm.pricing", (), ""),
    g.Edge("evals.grade", "llm.core", (), "types only"),
]
INTERNAL = ["tools", "tools_config", "tools.impl", "evals.runner", "evals.grade", "llm.pricing"]


@pytest.mark.parametrize(
    ("rule", "broken"),
    [
        pytest.param("- `evals` —never-imports→ `llm`", ["evals.runner"], id="never, broken"),
        pytest.param("- `evals` —never-imports→ `retrieval`", [], id="never, kept"),
        pytest.param("- `tools` —never-imports→ `numpy`", ["tools"], id="never, a package"),
        pytest.param("- `tools` —only-imports→ `tools_config`", [], id="only, own submodules"),
        pytest.param("- `tools` —only-imports→ nothing", ["tools"], id="only nothing, broken"),
        pytest.param("- `evals.grade` —only-imports→ nothing", [], id="types only is exempt"),
        pytest.param("- `evals` —only-imports→ `evals`, `llm`", [], id="only, a list"),
        pytest.param("tools never imports config", [], id="prose is no rule"),
    ],
)
def test_layering_rules_are_checked_against_the_edges(rule: str, broken: list[str]) -> None:
    errors = g.broken_rules(rule, EDGES, INTERNAL)
    assert [e.split(" ")[0] for e in errors] == broken, errors


def test_at_lines_are_placed_replaced_and_checked(resolver: g.Resolver) -> None:
    lines = [
        "#### `core.helper`",
        "- **At:** `src/pkg/core.py:1`",
        "- **Is:** function.",
        "#### `core.Llm`",
        "- **Is:** class.",
        "#### `core.gone`",
        "#### `docs/gone.md`",
        "#### `core.Llm`",
        "#### roadmap",
    ]
    out, errors = g.place_at_lines(lines, resolver)
    assert out[:6] == [
        "#### `core.helper`",
        "- **At:** `src/pkg/core.py:26`",
        "- **Is:** function.",
        "#### `core.Llm`",
        "- **At:** `src/pkg/core.py:17`",
        "- **Is:** class.",
    ]
    assert errors == [
        "node `core.gone` names nothing in the code (renamed or removed?)",
        "node `docs/gone.md` names a path that does not exist",
        "node `core.Llm` is defined twice",
    ]


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        pytest.param(["At", "Is", "Contract", "History", "Decided in"], None, id="in order"),
        pytest.param(
            ["Is", "Gotchas", "Invariants"], "field `Invariants` out of order", id="swapped"
        ),
        pytest.param(["Is", "Is"], "field `Is` out of order", id="repeated"),
        pytest.param(["Is", "Notes"], "unknown field `Notes`", id="unknown"),
    ],
)
def test_a_nodes_fields_keep_one_order(fields: list[str], error: str | None) -> None:
    lines = ["#### `x`", *(f"- **{name}:** text" for name in fields), "", "#### `y`", "- **Is:** y"]
    errors = g.misordered_fields(lines)
    assert errors == ([] if error is None else [f"node `x`: {error}"])


def test_fields_after_a_nodes_body_are_not_its() -> None:
    lines = ["#### `x`", "- **History:** x", "", "Prose.", "- **Is:** in a list, not a node"]
    assert g.misordered_fields(lines) == []


def test_the_contents_skip_headings_in_code_blocks() -> None:
    lines = ["## A", "```", "## not a heading", "```", "### B", "#### `c`"]
    assert g.render_toc(lines, 0) == ["- L1 A", "  - L5 B", "    - L6 `c`"]


GRAPH = """# Graph

## Contents

<!-- generated:toc (x) -->
stale
<!-- /generated:toc -->

## 2. Nodes

#### `core.helper`
- **Is:** function.

## 3. Edges

<!-- generated:imports (x) -->
<!-- /generated:imports -->

- `core` —never-imports→ `numpy`
"""


def test_generate_fills_every_part_and_reports_broken_rules(repo: Path) -> None:
    result = g.generate(GRAPH, repo, PACKAGE)
    lines = result.text.split("\n")
    toc = [line for line in lines if line.lstrip().startswith("- L")]
    for entry in toc:
        number, title = entry.strip()[3:].split(" ", 1)
        assert lines[int(number) - 1].endswith(title), entry
    assert "- **At:** `src/pkg/core.py:26`" in lines
    assert "core —imports→ config.{LIMIT, NAME}" in lines
    assert any("breaking: `core` —never-imports→ `numpy`" in e for e in result.errors)


def test_generate_is_idempotent(repo: Path) -> None:
    once = g.generate(GRAPH, repo, PACKAGE).text
    assert g.generate(once, repo, PACKAGE).text == once


@pytest.mark.parametrize("region", ["toc", "imports"])
def test_a_missing_region_is_an_error(repo: Path, region: str) -> None:
    text = GRAPH.replace(f"<!-- /generated:{region} -->", "")
    errors = g.generate(text, repo, PACKAGE).errors
    assert f"the graph has no `generated:{region}` region" in errors


@pytest.fixture
def graph_file(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = repo / "docs" / "graph.md"
    path.write_text(GRAPH.replace("- `core` —never-imports→ `numpy`", ""))
    monkeypatch.setattr(g, "ROOT", repo)
    monkeypatch.setattr(g, "GRAPH", path)
    monkeypatch.setattr(g, "PACKAGE", PACKAGE)
    return path


def test_check_fails_on_a_stale_file_and_writes_nothing(graph_file: Path) -> None:
    before = graph_file.read_text()
    assert g.main([]) == 1
    assert graph_file.read_text() == before


def test_write_then_check_passes(graph_file: Path) -> None:
    assert g.main(["--write"]) == 0
    assert g.main([]) == 0
