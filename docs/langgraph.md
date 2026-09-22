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
| a batch is admitted whole, before any of it is sent (`gather_*`) | **Bypassed.** A step's tasks run concurrently, 6 of 6 at once with no limit by default (*measured*), and each node's call is admitted alone. | `Budget` now holds calls in flight: see below |
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
each worker is its own task, admitted alone, while its siblings are in flight. The fix went into
`pricing.Budget`, not the graph: each admitted call's worst case is held until it is billed, so
admission counts calls in flight, and any framework's concurrency is safe under the same ceiling.

## The rebuild, compared

`workflows/orchestrator_workers_graph.py` rebuilds `workflows/orchestrator_workers.py` as a graph:
`plan`, then `gather`, then one `work` per subtask, then `synthesise`. It uses the same prompts,
the same plan schema and the same calls, all through `llm`. **The same eight tests pass against
both builds**: `test_orchestrator_workers` is one table run against each, so the rebuild is held to
behave exactly as the original.

**Lines.** The orchestration is 24 lines plain and 67 as a graph: `State` 7, the worker's `Brief`
4, `build` 48, `run` 8. The modules are 98 and 122 lines, and the graph's imports the prompts and
schema from the plain one.

**What the graph needed that the plain version did not:**

- a reducer on `findings`, because parallel workers write it in one step; and a number on each
  finding to restore plan order, because the reducer concatenates in completion order;
- a separate `gather` node, so searches stay one at a time; the workers run in threads, and the
  local search models are not shared across threads;
- `input_schema=Brief` on the worker node, and its argument named `state`: pyright holds a node to
  LangGraph's protocol by parameter name;
- a list of destinations on the conditional edge after `plan`. Without it, the drawn diagram showed
  the run ending at `plan`, with every node after it missing;
- an explicit `recursion_limit` (`MAX_GRAPH_STEPS`, 10) in place of the 10,007 default.

**Debugging**, measured with a worker made to fail and no model calls:

| | plain | graph |
|---|---|---|
| a failing worker | `RuntimeError`, 13 frames, 7 of them ours | the same error, 19 frames, 7 of them ours |
| calls billed before it surfaced | `plan`, `worker:1`, `worker:3` | the same |
| calls, tokens and cost | `llm.describe` | the same trace, unchanged |
| a picture of the flow | read `run` | `build(...).get_graph().draw_mermaid()`, once every conditional edge lists its destinations |
| state between steps | local variables | `stream_mode="updates"` per node; with a checkpointer, a checkpoint per step, and a resume that reruns only the worker that failed |

**One behaviour differs: admission.** The plain build admits all its workers as one batch, so if
the batch cannot fit, none is sent. The graph admits each worker as its task starts. Holds keep the
ceiling true, but a batch that does not fit is partly sent before the refusal. That changes how a
run fails, not how much it can spend.

**Verdict for this pattern:** the graph costs almost three times the orchestration lines, plus
rules the plain version never needed: reducers, input schemas, destination lists, a step limit.
The call trace, which is most of what debugging here needs, we already had. What it adds is real,
but for other shapes: checkpointed resume, a drawable flow, streaming per node, and interrupts for
human approval. Those pay off in a long-running agent with approvals and retries, which is
Week 4's Project 2, and not in a three-step workflow.

