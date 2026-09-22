"""Every error this package defines. Reuse one of these before adding another."""


class BudgetExceeded(RuntimeError):
    """Spending past a ceiling: refused before a call is sent, or raised once one has crossed it."""


class DatasetError(ValueError):
    """A dataset that cannot be trusted: bad JSON, an invalid row, or duplicate ids."""


class StepFailed(RuntimeError):
    """A call came back refused, truncated or unparseable. It was billed all the same."""


class TurnsExhausted(RuntimeError):
    """A tool-using conversation hit its turn cap with the model still asking for tools."""
