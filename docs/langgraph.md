# LangGraph, read against the loop we wrote

Week 3 asks for the quickstart and a read of the graph executor's source. This is `langgraph`
1.2.12 with `langgraph-checkpoint` 4.2.0, in the non-default `graph` dependency group. The core
loop is four files in `langgraph/pregel/` — `main.py` (4,364 lines), `_loop.py` (1,988), `_algo.py`
(1,460), `_runner.py` (941) — and `graph/state.py` (1,978) compiles a `StateGraph` down to it.
Every behaviour below marked *measured* was checked with a small graph that makes no model calls,
for $0.00.

## The loop, as the executor writes it

```
compile(StateGraph) → Pregel
    one channel per state key: LastValue, or BinaryOperatorAggregate when the key has a reducer
    each node subscribes to the channels that trigger it

while loop.tick():                       # _loop.py:599 — stop at recursion_limit
    tasks = prepare_next_tasks(...)      # nodes whose triggers changed last step, plus Send packets
    runner.tick(tasks)                   # every task of the step at once, in threads
    loop.after_tick()                    # apply_writes: reducers merge; a checkpoint is saved
```

It is bulk-synchronous: a **superstep** runs every ready node together, then merges their writes,
then decides the next step. Fan-in is automatic. In the quickstart, three parallel `work` tasks
were followed by one `approve`, not three (*measured*).

## The six features, as they actually run

| feature | what it is underneath |
|---|---|
| state | A `TypedDict`; each key is a channel. A key with no reducer takes **one write per step**. Two parallel nodes writing it raise `InvalidUpdateError` ("Can receive only one value per step"). A key annotated with a reducer, such as `operator.add`, merges them. |
| nodes | Functions from state to a partial update. An empty update streams as `None`. |
| conditional edges | A function of state that returns node names, or `Send` packets that fan out with their own input: orchestrator-workers in one line. |
| checkpointing | A checkpoint after every superstep, 6 for a 5-step run (*measured*); durability `"async"` by default, `"sync"` or `"exit"` on request. |
| interrupts | `interrupt(value)` raises `GraphInterrupt`, and `Command(resume=...)` continues. **The node reruns from its first line** (*measured*). Several interrupts in one node are matched to resume values by order. Needs a checkpointer. |
| streaming | Modes `values`, `updates`, `tasks`, `debug`, `messages`, `custom`, `checkpoints`. `updates` is one event per node; `tasks` shows what started in each step. |

## Against our rules

| our rule | LangGraph | |
|---|---|---|
| a loop is capped (invariant 6) | `recursion_limit` defaults to **10,007** supersteps, then raises `GraphRecursionError` (*measured*). It raises, unlike the tool runner, but with a model call a step that is ten thousand calls. | set it explicitly |
| every call is counted and admitted before it is sent (1, 22) | Knows nothing of cost. A node calls whatever it calls. | `llm` stays the door inside every node |
| a batch is admitted whole, before any of it is sent (`gather_*`) | **Bypassed.** A step's tasks run concurrently, 6 of 6 at once with no limit by default (*measured*). Each node's call is admitted alone, and `Budget.admit` does not count calls still in flight. | see below |
| tool errors are results, never exceptions (5) | Not ours to keep here: we write the nodes. The prebuilt `ToolNode` has its own rules. | — |

## What it hides

1. **The step cap is 10,007 by default.** Older LangChain material says 25, which is
   `langchain_core`'s default. LangGraph overrides it (`LANGGRAPH_DEFAULT_RECURSION_LIMIT`).
2. **An interrupt reruns its node from the top.** Anything before `interrupt()`, including a paid
   call, happens twice: once before the pause and once on resume. The docstring says so; a quick
   read of the quickstart would not.
3. **Parallel tasks are unbounded.** `max_concurrency` is off unless set in the run's config.
4. **A plain state key rejects parallel writes.** Parallel writers need a reducer, chosen when the
   state is declared.
5. **A checkpoint every superstep**, written asynchronously by default. That is cheap in memory
   and a write per step against a database.
6. **One kindness:** with a checkpointer, a finished task's writes are kept. When one of three
   parallel workers failed and the run was resumed, only the failed one ran again (*measured*),
   so paid calls that succeeded are not paid for twice.

## What it means for the rebuild

The rebuild of orchestrator-workers as a graph keeps every model call in `llm`, sets
`recursion_limit`, and keeps paid calls out of any node that interrupts. The one real conflict is
concurrency. The plain version admits all its workers as one batch before sending any; in a graph
each worker is its own task, admitted alone, while its siblings are in flight. The fix belongs in
`pricing.Budget`, not in the graph: reserve each admitted call's worst case until it is billed.
Admission then counts calls in flight, and any framework's concurrency is safe under the same
ceiling.
