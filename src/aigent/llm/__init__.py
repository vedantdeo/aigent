"""Everything that talks to a model: the door (`core`), the neutral types (`messages`), prices and
budgets (`pricing`), and the wires (`adapters`).

Import from the submodules. This file imports nothing: `config` imports `adapters` while `core`
imports `config`, so a re-export here would be a circular import.
"""
