# One agent, three frameworks

Project 2's agent built three times over the same prompt, tools and 25 tasks, graded by the same
eval (`aigent.agent_tasks`), every call through `llm`: the SDK's tool runner (`agent.py`),
LangGraph (`agent_graph.py`) and Google's ADK (`agent_adk.py`). CrewAI was the fourth and could
not be installed. Measured 2026-09-28; the runs and their costs are in `LOG.md`.

| | SDK tool runner | LangGraph | Google ADK | CrewAI |
|---|---|---|---|---|
| 25 tasks, Sonnet agent, Opus judge | not run | **23/25**, $0.038 a task | 20/25, $0.037 a task | — |
| Lines of agent code | 149 | 198 | 236 | — |
| Installs beside this repo | yes | yes (graph group) | yes, +24 packages | **no** |
| Calls go through our `llm` | yes (`run_tools`) | yes (we drive each turn) | yes, via `BaseLlm` | impossible |
| Claude's server-side web search | works | works | **drops it**: three workarounds | — |
| Tool schemas | ours | ours | **derived from docstrings** | — |
| Tool errors marked `is_error` | yes | yes | **no**, plain text | — |
| Workarounds found by live runs | — | 1 (`reply_of` nulls) | 3 (history replay) | — |

Read the scores with their spread: the same LangGraph agent scored 15 and 23 on consecutive runs
with four tasks flipping and nothing changed, so a three-task gap is inside one run's noise. The
cost per task is the same everywhere because every build sends the same requests through `llm`;
the framework adds orchestration, not tokens.

**Control.** The SDK runner hides the loop and gives no turn cap or budget by default; `llm` wraps
it (`docs/tool-runner.md`). LangGraph makes you draw the loop — a model node, a tools node, the
edges — which is more code, and every rule of ours (one message for all tool results, errors as
`is_error`, a last answer when out of room) is ours to write and therefore ours to keep. ADK runs
its own loop and exposes one seam, the model class; our rules survive only because that seam let us
route every call through `llm`, and a last answer had to be rebuilt inside it.

**Debuggability.** The deciding difference is **who owns the conversation**. In LangGraph the
message list is ours, sent exactly as the API returned it, so the API's rules about history hold
without effort. ADK keeps history in Google's own format and converts both ways each turn; its
Anthropic converter cannot represent a server tool, rewrites thinking as redacted, and cannot know
that a web result may arrive a turn after its call. Each of those surfaced only as a 400 from a
live run — the fake-client tests passed every time — and the last took a diagnostic run to see.

**Cost visibility.** Equal, and only because of `llm`: every build is admitted per turn, capped per
task and billed to one ledger. Out of the box, CrewAI would call Claude through its own SDK with an
iteration cap and a rate limit, and no pre-flight budget at all.

**Lock-in.** CrewAI pins `openai<3`, `pydantic<2.13` and `anthropic~=0.73` — below this repo's, in
every release — so using it means a second environment and a process boundary. ADK installs
cleanly but is built Gemini-first: Claude's own features pass through it lossily. LangGraph asks
the least of the model layer and so leaves the most in our hands.

**The answer.** For a Claude agent that must keep its own rules and its own ledger, LangGraph:
the loop is visible, the history is ours, and nothing between us and the API rewrites a message.
ADK earns its place when the model is Gemini or the team wants its runner and session machinery,
and it costs a translation layer that has to be tested against the live API. CrewAI's team
metaphor was not tested here; its dependency pins ended the experiment before the first call.
