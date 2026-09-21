# Failure modes

What this codebase does when things go wrong, provoked rather than described.
Regenerate with `uv run python -m entropic.primitives.failures > docs/failure-modes.md`.
Every row is free: a rejected request is never billed, and the last three never leave the machine.

| failure | provoked by | what you get |
|---|---|---|
| `bad-model` | a model name that does not exist | `NotFoundError` — model: claude-opus-4-9 |
| `bad-key` | credentials that are not ours | `AuthenticationError` — API key is invalid. |
| `deprecated-sampling` | `temperature` on a Claude 5 model | `BadRequestError` — `temperature` is deprecated for this model. |
| `unknown-field` | a field the API has never heard of | `BadRequestError` — nonsense_field: Extra inputs are not permitted |
| `max-tokens-too-high` | an output cap above the model's limit | `ValueError` — Streaming is required for operations that may take longer than 10 minutes. See https://platform.claude.com/docs/en/cli-sdks- … |
| `empty-messages` | a request with nothing to answer | `BadRequestError` — messages: at least one message is required |
| `context-too-long` | an input larger than the context window | `BudgetExceeded` — request could cost up to $1.5256 (300006 input tokens plus max_tokens=1024 on claude-opus-5), above the per-request ceiling of $0.2500. Trim the … |
| `per-request-ceiling` | one call worth more than the ceiling | `BudgetExceeded` — request could cost up to $10.0002 (2000000 input tokens plus max_tokens=8 on claude-opus-5), above the per-request ceiling of $0.2500. Trim the … |
| `per-run-ceiling` | a run that spends past its budget | `BudgetExceeded` — run has spent $3.0000, above the per-run ceiling of $0.0100. Raise ENTROPIC_MAX_USD_PER_RUN in .env if this was intended. |
| `run-admission` | a call that could carry a run past its budget | `BudgetExceeded` — run has spent $0.9500 and the next spending could cost up to $0.1074, past the per-run ceiling of $1.0000. Raise ENTROPIC_MAX_USD_PER_RUN in .env if … |
