"""The primitives Entropic is made of, each runnable on its own and wired into the CLI.

    first_call          one call, token count, cost
    streaming           thinking and text arriving separately
    structured_output   schema in, validated object out
    tool_loop           the agent loop, by hand
    chat                multi-turn, with a cost meter

Each sends through `entropic.llm`, which counts, admits and bills every call.
"""
