# caspr-harness-ui

Browser harness for the Caspr agent in `caspr-core`
(`src/app/research/agent/` — set `CASPR_CORE_DIR` to point elsewhere).

One live `Casper`, two ways to drive it:

| Mode | What it does |
| --- | --- |
| **flow** | A full turn through `get_processing_state`. Forwards every stream item — per-token assistant text, `updates` node hops, `values` snapshots, and every `custom` event. Handles both stopping points inline: the `ask_user` pick-list and the `report_config` popup. |
| **node** | Invoke **one** node, router, tool, helper or module function on its own, with inputs you edit as JSON in the page. Shows the return value, every stream event it emitted, timing, and the traceback if it raised. |

The **node path** strip under the header tracks where the run is: one chip per
node hop, the live one highlighted, elapsed seconds on each.

## The two places a run stops

**`ask_user`** interrupts: the run holds the thread open, and your next message —
a clicked option or anything you type — goes back in as that call's result, so
the same run carries on. The page needs no separate control for it.

**`report_config`** does not interrupt. It shows both tiers, parks the model's
`retrieve` arguments, and ends the turn, which is why you can keep chatting
while the popup is open (a message supersedes it). **confirm** replays those
arguments through `run_confirmed_report` on a thread of its own. Confirming the
tier the chat already refined retrieves; switching tier hands that layout back to
the chat through `layout_handoff` and generates nothing until you confirm again.

`respond_during_report` and the refine-request capturer are commented out in
core, so the report-in-progress controls no longer route a follow-up anywhere and
the refine panels stay empty. The harness imports the capturer optionally and
runs fine without it.

## User memory

Casper is built with the user memory in `$CASPR_USER_MEMORY`, defaulting to the
checkout's `tests/fixtures/sample_user_memory.json`. The memory sets the chat
tone and the default report audience (`profile.audience`), so the harness shows
the personalised Caspr, not the generic one. The header pill says which memory
loaded and the audience it carries; hover it for the path.

```bash
CASPR_USER_MEMORY=/path/to/my_memory.json uv run server.py   # another memory
CASPR_USER_MEMORY=off uv run server.py                       # no memory at all
```

A memory that is missing or not valid JSON is reported in the pill and Casper is
built without one, rather than failing the boot.

## Running against another checkout

```bash
CASPR_CORE_DIR=../caspr-core-old uv run server.py
```

A checkout that keeps the agent in one `model.py` (no `_chat` / `_graph` /
`_report` mixins) drops the old-only targets from node mode, leaving a smaller
catalog (`report_or_respond`, `propose_report_layout`, the read tools, gate and
routers). The harness sets `billing_is_external` either way: core checks the
local wallet before binding tools, and the harness user has none.

Its `propose_report_layout` streams like brief cards: the layout pane shows the
title as soon as the model's JSON opens `sections`, then each whole section the
moment its object closes (dashed while live), then the final layout replaces the
preview. The raw `propose_report_layout` custom events (`layout_stream_start`,
`layout_title`, `layout_section`, `layout_stream_complete`, `layout_stream_failed`)
stay visible in the event stream.

## Report generation is real

`generate_report`, `run_primary_research` and `run_due_diligence` are **not
stubbed**. A `retrieve` turn builds an actual report — minutes of work and real
LLM spend. Only unreachable infrastructure is muted (Postgres cost/analytics
writers, CloudWatch); the agent code itself runs untouched.

## Run

Needs a populated `.env` in the checkout. Any interpreter works — `server.py`
re-execs itself under that checkout's `.venv`:

```bash
cd caspr-harness-ui
uv run server.py            # or: python server.py
# open http://127.0.0.1:8765
```

Override with `HARNESS_HOST` / `HARNESS_PORT` / `CASPR_CORE_DIR`.

## Node mode

Targets are listed grouped by the mixin file they live in (`_chat`, `_retrieve`,
`_report`, `_graph`, `_helpers`), with a `node` / `router` / `tool` / `helper` /
`func` badge. Every signature and docstring is read off the live object at
connect time, so the list follows the code rather than drifting from it.

Inputs are JSON. Three placeholders are expanded server-side:

| Placeholder | Becomes |
| --- | --- |
| `"$HISTORY"` | the current flow-mode conversation, as LangChain messages |
| a `messages` list of `{type, content, …}` | `SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage` objects |
| `["$MESSAGE_DICTS", …]` | the same, then `.model_dump()`ed (what `_keep_finished_chat_history` takes) |

**prepend real system message** splices `casper.system_message` in front, so a
node sees the prompt it would see in a real run.

Nodes emit through `get_stream_writer()`, which normally only resolves inside a
graph run. For a single-node run the harness patches that name on every `app.*`
module that imported it, so events are captured and streamed to the page, then
restores it. `ask_user` is the one target that can't run standalone — it
calls `interrupt()` — and it's marked `graph` in the list. `report_config` runs
fine on its own now: it emits the request and returns.

## Controls

- **load scenario** — `report_in_progress` ON with the sample EV layout, so chat
  routes to `respond_during_report`. Then try *"add a paragraph on charging
  infrastructure to the Risks section"* and watch `post_report_edit_request`.
- **toggle report flag** — flips `report_in_progress` (entry routing).
- **reset** — fresh message history and a new turn thread, same `Casper`.
- `Enter` sends a chat message; `Ctrl/Cmd+Enter` runs the selected node.
