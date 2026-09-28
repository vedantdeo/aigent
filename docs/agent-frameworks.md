# One agent, three frameworks

Project 2's agent built three times over the same prompt, tools and 25 tasks, graded by the same
eval (`aigent.agent_tasks`), every call through `llm`: the SDK's tool runner (`agent.py`),
LangGraph (`agent_graph.py`) and Google's ADK (`agent_adk.py`). CrewAI was the fourth and could
not be installed. Measured 2026-09-28; the runs and their costs are in `LOG.md`.

| | SDK tool runner | LangGraph | Google ADK | CrewAI |
|---|---|---|---|---|
| 25 tasks, Sonnet agent, Opus judge | 16/25, $0.039 a task | 23/25, $0.038 a task | 20/25, $0.037 a task | — |
| Lines of agent code | 149 | 198 | 236 | — |
| Installs beside this repo | yes | yes (graph group) | yes, +24 packages | **no** |
| Calls go through our `llm` | yes (`run_tools`) | yes (we drive each turn) | yes, via `BaseLlm` | impossible |
| Claude's server-side web search | works | works | **drops it**: three workarounds | — |
| Tool schemas | ours | ours | **derived from docstrings** | — |
| Tool errors marked `is_error` | yes | yes | **no**, plain text | — |
| Workarounds found by live runs | — | 1 (`reply_of` nulls) | 3 (history replay) | — |

**For this agent the frameworks add nothing we did not already have.** The scores, 16 to 23, sit
inside the spread of one build on its own — the same LangGraph agent scored 15 and 23 on
consecutive runs — so the framework does not move accuracy, and the cost per task is the same
everywhere because every build sends the same requests through `llm`. Everything hard (admission,
ceilings, the last answer, caching, the ledger) was ours already and had to be routed *through*
each framework, not replaced by it. That flips only when the work needs what we have not built:
resuming a long task after a pause (LangGraph's checkpoints), several agents sharing state and
handing off, or hosted sessions and memory (ADK).

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

**The answer.** For this agent, our own loop: `llm.run_tools` on the SDK's runner is the least
code, scored inside the others' spread, and keeps every rule without a translation layer. Reach for
LangGraph when the work needs checkpoints to resume, human approval mid-run, or several agents
sharing state — it is the framework that leaves the message list in our hands, so it costs least to
adopt behind `llm`. ADK earns its place when the model is Gemini or the team wants its hosted
sessions, and it costs a translation layer that must be tested against the live API. CrewAI's team
metaphor was not tested; its dependency pins ended the experiment before the first call.

**After the comparison.** Web answers in every build lost marks for naming no source: the API's
structured citations sit beside the text, where the judge never saw them. Two fixes followed, both
build-independent. The judge now reads the cited pages after the answer, and the shared prompt asks
for the site and date beside every web figure. The prompt is the one that moved results: on
LangGraph's three web tasks, 2 of 3 answers now name their source in the text, and those two pass
(LOG, 2026-09-28). The scores above predate both, so they are comparable with each other, not with
a run after.
