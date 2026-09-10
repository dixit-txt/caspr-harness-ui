# caspr-harness-ui

Browser harness for the Caspr agent in `caspr-core-old`
(`src/app/research/agent/` — set `CASPR_CORE_DIR` to point elsewhere).

One live `Casper`, two ways to drive it:

| Mode | What it does |
| --- | --- |
| **flow** | A full turn through `get_processing_state`. Forwards every stream item — per-token assistant text, `updates` node hops, `values` snapshots, and every `custom` event. Handles the `ask_user` pick-list and the `report_config` pause/resume inline. |
| **node** | Invoke **one** node, router, tool, helper or module function on its own, with inputs you edit as JSON in the page. Shows the return value, every stream event it emitted, timing, and the traceback if it raised. |

The **node path** strip under the header tracks where the run is: one chip per
node hop, the live one highlighted, elapsed seconds on each.

## Report generation is real

`generate_report`, `run_primary_research` and `run_due_diligence` are **not
stubbed**. A `retrieve` turn builds an actual report — minutes of work and real
LLM spend. Only unreachable infrastructure is muted (Postgres cost/analytics
writers, CloudWatch); the agent code itself runs untouched.

## Run

Needs a populated `.env` in `caspr-core-old`. Any interpreter works — `server.py`
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
restores it. `report_config` is the one target that can't run standalone — it
calls `interrupt()` — and it's marked `graph` in the list.

## Controls

- **load scenario** — `report_in_progress` ON with the sample EV layout, so chat
  routes to `respond_during_report`. Then try *"add a paragraph on charging
  infrastructure to the Risks section"* and watch `post_report_edit_request`.
- **toggle report flag** — flips `report_in_progress` (entry routing).
- **reset** — fresh message history and a new turn thread, same `Casper`.
- `Enter` sends a chat message; `Ctrl/Cmd+Enter` runs the selected node.
