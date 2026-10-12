# Developing aigent

How the repo is checked, released and edited. The [README](../README.md) is what aigent does; this
is how it stays working.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest -q            # unit tests plus a free API smoke test; skips paid tests
```

A paid test carries `@pytest.mark.live` and runs only with `uv run pytest -m live`, so the default
suite cannot spend anything.

Everything above runs in CI, in two pipelines over one set of steps
(`.github/workflows/checks.yml`): the **mains CI** (`main.yml`) on every push to `main` and every
pull request, against the commit underhood's `main` is at, and started again whenever underhood's own
`main` goes green; and the **roll** jobs in `release.yml` on each new pair of snaps. No API key is
configured there, so the smoke test skips itself — CI spends nothing. `uv sync --locked` fails the
run if `uv.lock` has drifted from `pyproject.toml`.

`docs/knowledge-graph.md` maps the repo, and it moves with the code: a commit that touches `src/` or
`pyproject.toml` without staging the graph is refused by `.githooks/pre-commit` (enable it with
`git config core.hooksPath .githooks`, since git never installs hooks on clone), and the same rule
runs in CI for clones that skipped the hook. The hook also runs `ruff check` and `ruff format
--check` over the staged files.

`main` takes changes only through pull requests, in both repos: a ruleset refuses direct and force
pushes and waits for the checks above.

## Snap and roll

aigent and underhood are released as a pair, all or nothing, by `.github/workflows/release.yml`.
Twice a week, at Mon and Thu 00:07 IST (Sun and Wed 18:37 UTC; an odd minute, since GitHub runs
schedules late at busy ones):

- **snap** branches underhood's `main` as `snap-YYYY-MM-DD`, then branches aigent's `main` the same
  way and re-pins only that branch to underhood's snap. Each repo keeps its last eight snaps.
- **roll** runs both repos' checks at once: underhood's own `checks.yml` on its snap, and aigent's on
  its snap against the underhood it pins.
- **stable**, only if both pass, tags both snaps `roll-YYYY-MM-DD` and moves each repo's `stable` tag
  there. Otherwise nothing moves in either repo and the run goes red.
- So the mains pair with each other and the snaps with each other: aigent's `main` always pins
  underhood's `main`, and a snap never touches it.
- **A fix after a red roll** lands on `main` first, then is cherry-picked onto the snap branch. Run
  the workflow by hand with step `roll` to roll the latest pair again, or, after an underhood fix,
  `repin`, which re-pins aigent's snap to underhood's snap head, then rolls.
- `UNDERHOOD_TOKEN` (Contents read and write on underhood) lets it branch and tag underhood.

## Editor

`.vscode/` is committed, so the settings arrive with the clone; VS Code offers the extensions on
first open (Ruff, Python, Pylance).

| On save | Does what |
|---------|-----------|
| `ruff format` | whitespace and layout, to the 100-char limit in `pyproject.toml` |
| `source.fixAll.ruff` | every safe autofix, including removing unused imports |
| `source.organizeImports.ruff` | sorts imports — the formatter does not do this |

`ruff.importStrategy` is `fromEnvironment`, so the editor runs `.venv/bin/ruff` and agrees with
`uv run ruff check`. Run `uv sync` before opening the folder. The test runner inherits
`addopts = "-m 'not live'"`, so running the whole suite from the editor cannot spend anything.

## Working with Claude Code

`CLAUDE.md` holds the project rules every Claude Code session follows here — tests ship with
features, paid calls need per-action approval, constants live in one config module, the knowledge
graph moves with every commit. They are written down so they travel with the repo.
