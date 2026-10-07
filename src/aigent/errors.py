"""Every error this package defines. Reuse one of these before adding another.

Classes stay in alphabetical order by name, so a new error has one place to go.
"""


class BatchUnfinished(RuntimeError):
    """A submitted batch that could not be collected; its requests may still be billed."""


class BudgetExceeded(RuntimeError):
    """Spending past a ceiling: refused before a call is sent, or raised once one has crossed it."""


class DatasetError(ValueError):
    """A dataset that cannot be trusted: bad JSON, an invalid row, or duplicate ids."""


class GuardrailTripped(ValueError):
    """Input a guardrail refused before anything was sent."""


class StepFailed(RuntimeError):
    """A call came back refused, truncated or unparseable. It was billed all the same."""


class TurnsExhausted(RuntimeError):
    """A tool-using conversation hit its turn cap with the model still asking for tools."""


class Unreadable(RuntimeError):
    """A reply for a record that did not validate, raised before its usage was seen."""


class Unsupported(RuntimeError):
    """A request asks for something the provider it is bound for cannot do."""
