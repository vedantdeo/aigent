"""The primitives Entropic is made of, each runnable on its own and wired into the CLI.

    first_call          one call, token count, cost
    streaming           thinking and text arriving separately
    structured_output   schema in, validated object out
    tool_loop           the agent loop, by hand
    chat                multi-turn, with a cost meter

`failures` is not a mode: it provokes nine failure modes and writes `docs/failure-modes.md`.
"""
