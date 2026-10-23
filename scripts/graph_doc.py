"""Keeps the generated parts of docs/knowledge-graph.md current: its contents, each node's `At:`
line, the import edges, and the `—never-imports→` / `—only-imports→` rules checked against the code.

    uv run python scripts/graph_doc.py           check; exit 1 if stale or a rule is broken
    uv run python scripts/graph_doc.py --write   rewrite the generated parts, then check

Standard library only, so CI needs Python 3.12 but not the project's dependencies.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "docs" / "knowledge-graph.md"
PACKAGE = "aigent"

NODE = re.compile(r"^#### `([^`]+)`")
SECTION = re.compile(r"^(#{2,4}) (.+)$")
AT_LINE = re.compile(r"^- \*\*At:\*\* ")
RULE = re.compile(r"^- `([\w.]+)` —(never|only)-imports→ (nothing|`[\w.]+`(?:, `[\w.]+`)*)")
PATHLIKE = re.compile(r"[/]|\.(?:py|md|json|jsonl|toml|yml|yaml|sh|txt|sqlite)$")
CODE_ID = re.compile(r"^[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*$")
FIELDS = ("At", "Is", "Contract", "Invariants", "Gotchas", "History", "Named at", "Decided in")
FIELD = re.compile(r"^- \*\*([^*]+):\*\* ")


@dataclass(frozen=True)
class Edge:
    """One module importing from another; `how` is "", "inside a function" or "types only"."""

    subject: str
    target: str
    names: tuple[str, ...]
    how: str


@dataclass
class Result:
    text: str
    errors: list[str] = field(default_factory=list)


def modules_under(root: Path, prefix: str) -> dict[str, Path]:
    """Dotted module name → file, for every `.py` under `root`; `prefix` heads each name."""
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join([prefix, *parts] if prefix else parts)
        if name:
            found[name] = path
    return found


def symbols_in(path: Path) -> dict[str, int]:
    """Top-level and class-level names, re-exports included → the line each is bound on."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: dict[str, int] = {}

    def visit(body: Sequence[ast.stmt], prefix: str) -> None:
        for stmt in body:
            names: list[str] = []
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [stmt.name]
            elif isinstance(stmt, ast.Assign):
                names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                names = [stmt.target.id]
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)) and not prefix:
                names = [(a.asname or a.name).split(".")[0] for a in stmt.names]
            for name in names:
                found.setdefault(prefix + name, stmt.lineno)
            if isinstance(stmt, ast.ClassDef):
                visit(stmt.body, f"{prefix}{stmt.name}.")

    visit(tree.body, "")
    return found


class Resolver:
    """Turns a node id into `path` or `path:line`, caching each module's symbols."""

    def __init__(self, root: Path, modules: Mapping[str, Path]) -> None:
        self.root = root
        self.modules = modules
        self.tops = {name.split(".")[0] for name in modules}
        self._symbols: dict[Path, dict[str, int]] = {}

    def is_code_id(self, node_id: str) -> bool:
        return bool(CODE_ID.match(node_id)) and node_id.split(".")[0] in self.tops

    def at(self, node_id: str) -> str | None:
        if PATHLIKE.search(node_id):
            return node_id if self._exists(node_id) else None
        if not self.is_code_id(node_id):
            return None
        parts = node_id.split(".")
        for cut in range(len(parts), 0, -1):
            module = ".".join(parts[:cut])
            path = self.modules.get(module)
            if path is None:
                continue
            rel = path.relative_to(self.root).as_posix()
            rest = ".".join(parts[cut:])
            if not rest:
                return rel
            if path not in self._symbols:
                self._symbols[path] = symbols_in(path)
            line = self._symbols[path].get(rest)
            return f"{rel}:{line}" if line is not None else None
        return None

    def is_missing_path(self, node_id: str) -> bool:
        return bool(PATHLIKE.search(node_id)) and not self._exists(node_id)

    def _exists(self, rel: str) -> bool:
        if re.search(r"[*<{ ]", rel):
            return any(self.root.glob(re.sub(r"<[^>]*>|\{[^}]*\}", "*", rel).split(" ")[0]))
        return (self.root / rel.rstrip("/")).exists()


def imports_of(modules: Mapping[str, Path], package: str) -> list[Edge]:
    """Every non-stdlib import in `modules`, internal targets named without the package prefix."""
    internal = set(modules)
    edges: set[Edge] = set()
    for subject, path in modules.items():
        tree = ast.parse(path.read_text(), filename=str(path))
        is_package = path.name == "__init__.py"
        for node, how in _walk_imports(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = _local(alias.name, package)
                    if target is not None or not _stdlib(alias.name):
                        edges.add(Edge(subject, target or alias.name, (), how))
                continue
            base = _absolute(node, subject, is_package, package)
            if base is None:
                continue
            target = _local(base, package)
            if target is None:
                if not _stdlib(base):
                    names = tuple(sorted(a.name for a in node.names))
                    edges.add(Edge(subject, base, names, how))
                continue
            plain: list[str] = []
            for alias in node.names:
                sub = f"{target}.{alias.name}" if target else alias.name
                if sub in internal:
                    edges.add(Edge(subject, sub, (), how))
                else:
                    plain.append(alias.name)
            if plain and target:
                edges.add(Edge(subject, target, tuple(sorted(plain)), how))
    merged: dict[tuple[str, str, str], set[str]] = {}
    for edge in edges:
        merged.setdefault((edge.subject, edge.target, edge.how), set()).update(edge.names)
    return [Edge(s, t, tuple(sorted(n)), h) for (s, t, h), n in sorted(merged.items())]


def _walk_imports(tree: ast.Module) -> Iterable[tuple[ast.Import | ast.ImportFrom, str]]:
    def visit(node: ast.AST, how: str) -> Iterable[tuple[ast.Import | ast.ImportFrom, str]]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, how
            return
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for stmt in node.body:
                yield from visit(stmt, "types only")
            for stmt in node.orelse:
                yield from visit(stmt, how)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and how != "types only":
            how = "inside a function"
        for child in ast.iter_child_nodes(node):
            yield from visit(child, how)

    yield from visit(tree, "")


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _absolute(node: ast.ImportFrom, subject: str, is_package: bool, package: str) -> str | None:
    if node.level == 0:
        return node.module
    parts = [package, *subject.split(".")] if subject else [package]
    if not is_package:
        parts = parts[:-1]
    if node.level > 1:
        parts = parts[: len(parts) - (node.level - 1)]
    if not parts:
        return None
    return ".".join([*parts, node.module] if node.module else parts)


def _local(name: str, package: str) -> str | None:
    if name == package:
        return ""
    return name[len(package) + 1 :] if name.startswith(package + ".") else None


def _stdlib(name: str) -> bool:
    return name.split(".")[0] in sys.stdlib_module_names or name == "__future__"


def render_imports(edges: Sequence[Edge]) -> str:
    """One line per (subject, target, how), so a grep for either end lands on it."""
    lines: list[str] = []
    for edge in edges:
        target = edge.target or PACKAGE
        if len(edge.names) == 1:
            target += f".{edge.names[0]}"
        elif edge.names:
            target += ".{" + ", ".join(edge.names) + "}"
        note = f"   ({edge.how})" if edge.how else ""
        lines.append(f"{edge.subject or PACKAGE} —imports→ {target}{note}")
    return "```\n" + "\n".join(lines) + "\n```"


def broken_rules(text: str, edges: Sequence[Edge], internal: Iterable[str]) -> list[str]:
    """Each `—never-imports→` or `—only-imports→` rule in the graph that a runtime import breaks."""
    inside = set(internal)
    errors: list[str] = []
    for line in text.splitlines():
        rule = RULE.match(line)
        if rule is None:
            continue
        importer, kind, listed = rule.groups()
        targets = re.findall(r"`([\w.]+)`", listed)
        for edge in edges:
            if edge.how == "types only" or not _under(edge.subject, importer):
                continue
            named = any(_under(edge.target, t) for t in targets)
            if kind == "never" and named:
                errors.append(f"{edge.subject} imports {edge.target}, breaking: {line[2:]}")
            elif kind == "only" and edge.target in inside and not named:
                if not _under(edge.target, importer):
                    errors.append(f"{edge.subject} imports {edge.target}, breaking: {line[2:]}")
    return errors


def _under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def place_at_lines(lines: list[str], resolver: Resolver) -> tuple[list[str], list[str]]:
    """Rewrites the `At:` line under every node heading; errors name duplicated or missing ids."""
    out: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        node = NODE.match(line)
        if node is None:
            continue
        node_id = node.group(1)
        if node_id in seen:
            errors.append(f"node `{node_id}` is defined twice")
        seen.add(node_id)
        if i < len(lines) and AT_LINE.match(lines[i]):
            i += 1
        at = resolver.at(node_id)
        if at is not None:
            out.append(f"- **At:** `{at}`")
        elif resolver.is_code_id(node_id):
            errors.append(f"node `{node_id}` names nothing in the code (renamed or removed?)")
        elif resolver.is_missing_path(node_id):
            errors.append(f"node `{node_id}` names a path that does not exist")
    return out, errors


def misordered_fields(lines: Sequence[str]) -> list[str]:
    """Each node whose field lines are unknown, repeated or out of `FIELDS` order."""
    errors: list[str] = []
    heading: str | None = None
    last = -1
    for line in lines:
        if line.startswith("#### "):
            heading, last = line[5:], -1
            continue
        if heading is None:
            continue
        field = FIELD.match(line)
        if field is None:
            if not line.startswith("  "):
                heading = None
            continue
        name = field.group(1)
        if name not in FIELDS:
            errors.append(f"node {heading}: unknown field `{name}`")
        elif FIELDS.index(name) <= last:
            errors.append(f"node {heading}: field `{name}` out of order")
        else:
            last = FIELDS.index(name)
    return errors


def render_toc(lines: Sequence[str], start: int) -> list[str]:
    """Every heading from line `start` (0-based) on, with its 1-based line number."""
    toc: list[str] = []
    fenced = False
    for number, line in enumerate(lines[start:], start=start + 1):
        if line.startswith("```"):
            fenced = not fenced
        heading = None if fenced else SECTION.match(line)
        if heading is None:
            continue
        depth = len(heading.group(1)) - 2
        toc.append(f"{'  ' * depth}- L{number} {heading.group(2)}")
    return toc


def region(lines: Sequence[str], name: str) -> tuple[int, int] | None:
    """The 0-based line indexes of a region's opening and closing markers."""
    begin = f"<!-- generated:{name}"
    end = f"<!-- /generated:{name} -->"
    opened = next((i for i, line in enumerate(lines) if line.startswith(begin)), None)
    if opened is None:
        return None
    closed = next((i for i in range(opened + 1, len(lines)) if lines[i] == end), None)
    return None if closed is None else (opened, closed)


def replace_region(lines: list[str], name: str, body: Sequence[str]) -> list[str] | None:
    found = region(lines, name)
    if found is None:
        return None
    opened, closed = found
    return [*lines[: opened + 1], *body, *lines[closed:]]


def generate(text: str, root: Path, package: str = PACKAGE) -> Result:
    """The graph with every generated part rewritten, and what the check found wrong."""
    src = modules_under(root / "src" / package, "")
    tests = modules_under(root / "tests", "tests")
    resolver = Resolver(root, {**src, **tests})
    edges = imports_of(src, package)
    errors = broken_rules(text, edges, src)

    lines = text.split("\n")
    lines, at_errors = place_at_lines(lines, resolver)
    errors += at_errors + misordered_fields(lines)

    for name, body in (("imports", render_imports(edges).split("\n")), ("toc", [])):
        replaced = replace_region(lines, name, body)
        if replaced is None:
            errors.append(f"the graph has no `generated:{name}` region")
        else:
            lines = replaced

    found = region(lines, "toc")
    if found is not None:
        opened, closed = found
        after = closed + 1
        # The contents' own length shifts every line below it, so measure once and fill in.
        toc = render_toc(lines, after)
        lines = [*lines[: opened + 1], *toc, *lines[closed:]]
        shift = len(toc)
        toc = render_toc(lines, after + shift)
        lines = [*lines[: opened + 1], *toc, *lines[opened + 1 + shift :]]
    return Result("\n".join(lines), errors)


def main(argv: Sequence[str]) -> int:
    write = "--write" in argv
    current = GRAPH.read_text()
    result = generate(current, ROOT, PACKAGE)
    if write and result.text != current:
        GRAPH.write_text(result.text)
        print(f"rewrote the generated parts of {GRAPH.relative_to(ROOT)}")
    stale = not write and result.text != current
    if stale:
        print(
            f"{GRAPH.relative_to(ROOT)}: generated parts are stale "
            "(contents, At: lines or import edges). "
            "Fix: uv run python scripts/graph_doc.py --write",
            file=sys.stderr,
        )
    for error in result.errors:
        print(f"{GRAPH.relative_to(ROOT)}: {error}", file=sys.stderr)
    return 1 if stale or result.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
