#!/usr/bin/env python3
"""Caspr agent harness — run the whole graph, or any single node/tool on its own.

Two modes over one WebSocket, against ONE live ``Casper`` built from the real
code in ``caspr-core-old`` (``$CASPR_CORE_DIR`` to override):

  flow   drive a full turn through ``get_processing_state`` and forward every
         stream item — per-token text, ``updates`` node hops, ``values``
         snapshots and every ``custom`` event — while tracking the node path.
         Report generation is REAL: ``generate_report`` / ``run_primary_research``
         / ``run_due_diligence`` are not stubbed, so a retrieve turn builds an
         actual report (slow, real LLM spend).

  node   invoke one target — a graph node, a router, a tool, a helper, or a
         module-level function — directly with inputs you edit in the page, and
         see its return value, every custom event it emitted, and its traceback
         if it raised. Targets are discovered from the live classes, grouped by
         the mixin file they live in, so the list follows the code.

Only the infrastructure the harness has no access to is neutralised (Postgres
cost/analytics writers, CloudWatch); the agent code itself runs untouched.

Needs a populated ``.env`` in the caspr checkout. Any interpreter works — this
re-execs itself under that checkout's ``.venv``:

    uv run server.py                       # or: python server.py
    # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── locate the caspr codebase this harness runs against ────────────────────
_HARNESS_DIR = Path(__file__).resolve().parent
_CASPR_CORE = Path(
    os.environ.get("CASPR_CORE_DIR", str(_HARNESS_DIR.parent / "caspr-core-old"))
).resolve()
_AGENT_DIR = _CASPR_CORE / "src" / "app" / "research" / "agent"

if not (_AGENT_DIR / "model.py").exists():
    print(
        f"  Casper agent package not found at {_AGENT_DIR}\n"
        f"  Set CASPR_CORE_DIR to the caspr checkout that has it.",
        file=sys.stderr,
    )
    raise SystemExit(1)

# ── run under caspr-core's venv, whatever interpreter launched us ──────────
# Compare venv roots, not interpreter realpaths — every venv's `python` symlinks
# back to the same system CPython, so realpath(sys.executable) can't tell them
# apart. `sys.prefix` is the venv dir itself.
_VENV = _CASPR_CORE / ".venv"
_VENV_PYTHON = _VENV / "bin" / "python"
if (
    _VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != _VENV.resolve()
    and not os.environ.get("_CASPR_HARNESS_UI_REEXEC")
):
    os.environ["_CASPR_HARNESS_UI_REEXEC"] = "1"
    os.environ["VIRTUAL_ENV"] = str(_VENV)
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), os.path.abspath(__file__), *sys.argv[1:]])

import asyncio  # noqa: E402
import contextlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from uuid import uuid7  # noqa: E402

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ── silence infrastructure the harness has no access to ────────────────────
# The app's fire-and-forget writers (cost tracker, analytics) target Postgres /
# CloudWatch / S3, none of which exist here. Their failures are noise, not
# signal — drop the log records and the asyncio task errors that mention them.
_NOISE_MARKERS = (
    "costtracker", "cost tracker", "cost_tracker", "insert_cost",
    "5432", "asyncpg", "ConnectionRefused", "Connect call failed",
    "cloudwatch", "CloudWatch", "greenlet", "MissingGreenlet",
    "boto", "endpoint", "credentials",
    "Database session", "async_session_scope", "TargetServerAttributeNotMatched",
    # a reachable-but-incompatible Redis answers HELLO with an error; the app
    # only uses it for session streaming, which this harness does not drive.
    "redis", "Redis", "6379", "unknown command",
)


class _DropInfraNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            text = record.getMessage()
        except Exception:
            text = str(getattr(record, "msg", ""))
        if record.exc_info and record.exc_info[1]:
            text += " " + repr(record.exc_info[1])
        return not any(m in text for m in _NOISE_MARKERS)


_DROP_FILTER = _DropInfraNoise()


def _install_global_log_filter() -> None:
    """Drop infra-noise records at the handler, not the logger.

    Adding the filter to each logger only covers the ones that exist when we
    look; the app creates several lazily on first use, and their records were
    still reaching stderr. Patching ``Handler.handle`` catches every record from
    every logger, whenever it is created.
    """
    if getattr(logging.Handler, "_caspr_harness_filtered", False):
        return
    original = logging.Handler.handle

    def handle(self, record):
        if not _DROP_FILTER.filter(record):
            return None
        return original(self, record)

    logging.Handler.handle = handle
    logging.Handler._caspr_harness_filtered = True


def _quiet_app_logging() -> None:
    """Call AFTER importing the app — ``setup_logging`` resets levels at import."""
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    _install_global_log_filter()
    logging.getLogger().addFilter(_DROP_FILTER)
    if not isinstance(sys.stderr, _FilteredStderr):
        sys.stderr = _FilteredStderr(sys.stderr)
    for name in ["", *list(logging.root.manager.loggerDict)]:
        lg = logging.getLogger(name)
        lg.addFilter(_DROP_FILTER)
        for h in list(getattr(lg, "handlers", [])):
            h.addFilter(_DROP_FILTER)
        if lg.level and lg.level < logging.WARNING:
            lg.setLevel(logging.WARNING)
    for loud in (
        "app.observability.llm_response_logger",
        "app.observability.web_search_analytics",
        "app.observability.error_alerter",
        "app.observability.cloudwatch_utils",
    ):
        logging.getLogger(loud).setLevel(logging.CRITICAL)


class _FilteredStderr:
    """Drop whole traceback blocks about unreachable infrastructure.

    Some app paths (``ErrorAlertManager.queue_error``) print straight to stderr
    rather than through ``logging``, so no log filter can reach them. Tracebacks
    are buffered as a block and judged as a whole — a marker on the exception
    line still suppresses the frames above it.
    """

    def __init__(self, wrapped):
        self._w = wrapped
        self._buf: list[str] = []
        self._in_tb = False

    def write(self, s):
        for line in str(s).splitlines(keepends=True):
            if line.startswith("Traceback (most recent call last)"):
                self._flush_block()
                self._in_tb = True
                self._buf.append(line)
            elif self._in_tb:
                self._buf.append(line)
                if line.strip() and not line[0].isspace():
                    self._in_tb = False       # the exception line closes the block
                    self._flush_block()
            elif not any(m in line for m in _NOISE_MARKERS):
                self._w.write(line)
        return len(str(s))

    def _flush_block(self):
        if not self._buf:
            return
        blob, self._buf = "".join(self._buf), []
        if not any(m in blob for m in _NOISE_MARKERS):
            self._w.write(blob)

    def flush(self):
        self._flush_block()
        self._w.flush()

    def isatty(self):
        return self._w.isatty()

    def fileno(self):
        return self._w.fileno()


def _swallow_infra_noise(loop, context):
    exc = context.get("exception")
    if isinstance(exc, (ConnectionRefusedError, OSError)):
        return
    blob = f"{context.get('message', '')} {exc!r}"
    if any(m in blob for m in _NOISE_MARKERS):
        return
    loop.default_exception_handler(context)


logging.disable(logging.INFO)

# ── import the app ─────────────────────────────────────────────────────────
sys.path.insert(0, str(_CASPR_CORE / "src"))

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.research.agent import _chat, _graph, _helpers, _report, _retrieve, _state  # noqa: E402
from app.research.agent.refine_request_router import (  # noqa: E402
    ROUTED_EVENT_NAME,
    RefineRequestRouter,
)
from app.research.agent.model import Casper  # noqa: E402

_quiet_app_logging()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

app = FastAPI(title="caspr agent harness")
app.mount("/static", StaticFiles(directory=str(_HARNESS_DIR / "static")), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_HARNESS_DIR / "static" / "index.html"))


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "caspr_core": str(_CASPR_CORE)})


@app.on_event("startup")
async def _on_startup() -> None:
    # Fire-and-forget writers raise into the loop, not into a request.
    asyncio.get_running_loop().set_exception_handler(_swallow_infra_noise)


# ── JSON-safe coercion ─────────────────────────────────────────────────────
def _safe(obj, _depth: int = 0):
    """Best-effort convert anything the graph produces into JSON-serialisable data."""
    if _depth > 12:
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    if hasattr(obj, "model_dump"):
        try:
            return _safe(obj.model_dump(), _depth + 1)
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _safe(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe(v, _depth + 1) for v in obj]
    return str(obj)


def _chunk_text(chunk) -> str:
    """Plain text of a message / message-chunk (Anthropic returns content blocks)."""
    c = getattr(chunk, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for part in c:
            if isinstance(part, dict) and part.get("type") in (None, "text") and "text" in part:
                out.append(part["text"])
        return "".join(out)
    return ""


def _dump_msgs(payload):
    """Turn a ``{'messages': [...]}`` node output into plain dicts for raw display."""
    if isinstance(payload, dict) and "messages" in payload:
        out = []
        for m in payload["messages"]:
            if hasattr(m, "model_dump"):
                d = m.model_dump()
                out.append(
                    {
                        k: d[k]
                        for k in ("type", "name", "content", "tool_calls", "tool_call_id")
                        if k in d and d[k] not in (None, [], "")
                    }
                )
            else:
                out.append(m)
        return {"messages": out}
    return payload


def _msg_digest(m, content_cap: int = 6000) -> dict:
    """Compact view of one message for the history / values feeds."""
    if hasattr(m, "model_dump"):
        d = m.model_dump()
    elif isinstance(m, dict):
        d = m
    else:
        d = {"content": str(m)}
    content = _chunk_text(m) if not isinstance(m, dict) else d.get("content", "")
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    content = content or ""
    out = {
        "type": d.get("type"),
        "name": d.get("name"),
        "content": content[:content_cap],
        "truncated": len(content) > content_cap,
    }
    if d.get("tool_calls"):
        out["tool_calls"] = _safe(d["tool_calls"])
    if d.get("tool_call_id"):
        out["tool_call_id"] = d["tool_call_id"]
    return out


def _mk_messages(specs, system: SystemMessage | None = None) -> list:
    """Materialise `[{type, content, ...}]` specs into LangChain message objects."""
    out: list = []
    if system is not None:
        out.append(system)
    for s in specs or []:
        if isinstance(s, str):
            out.append(HumanMessage(content=s))
            continue
        if not isinstance(s, dict):
            continue
        kind = (s.get("type") or "human").lower()
        content = s.get("content", "")
        if kind == "system":
            out.append(SystemMessage(content=content))
        elif kind == "ai":
            kwargs = {}
            if s.get("tool_calls"):
                kwargs["tool_calls"] = s["tool_calls"]
            out.append(AIMessage(content=content, **kwargs))
        elif kind == "tool":
            out.append(
                ToolMessage(
                    content=content,
                    tool_call_id=s.get("tool_call_id") or "call_harness_1",
                    name=s.get("name") or "",
                )
            )
        else:
            out.append(HumanMessage(content=content))
    return out


# ── event capture for standalone node runs ─────────────────────────────────
# Nodes emit through `get_stream_writer()`, which only resolves inside a live
# graph run. Every agent module holds its own imported reference, so patching
# the name on each module is what lets a node run on its own and still be
# observed — the same seam the unit tests use.
def _patchable_modules():
    mods = [_chat, _retrieve, _report, _graph]
    for name, mod in list(sys.modules.items()):
        if not name.startswith("app."):
            continue
        if mod is None or mod in mods:
            continue
        if hasattr(mod, "get_stream_writer"):
            mods.append(mod)
    return mods


@contextlib.contextmanager
def capture_stream_events(sink):
    """Route every ``get_stream_writer()`` call in the app to ``sink`` for the block."""

    def factory():
        return sink

    saved = []
    for mod in _patchable_modules():
        if hasattr(mod, "get_stream_writer"):
            saved.append((mod, mod.get_stream_writer))
            mod.get_stream_writer = factory
    try:
        yield
    finally:
        for mod, original in saved:
            mod.get_stream_writer = original


# ── target registry ────────────────────────────────────────────────────────
# Curated so the page lists things worth running, but every signature and
# docstring is read off the live object, so the list follows the code.
#
# kind:  node   -> takes `state`, returns a state update
#        router -> takes `state`, returns the next node's name
#        tool   -> the LLM-facing tools, called with keyword args
#        helper -> bound helpers worth poking at directly
#        func   -> module-level functions (no `self`)
_STATE_ARG = {"state": {"messages": "$MESSAGES"}}

_REGISTRY: list[dict] = [
    # ---- _chat.py ----
    {"file": "_chat", "kind": "node", "name": "report_or_respond", "args": _STATE_ARG,
     "note": "Main planner. Real LLM call; may emit tool calls."},
    {"file": "_chat", "kind": "node", "name": "respond_during_report", "args": _STATE_ARG,
     "note": "Chat-only node used while a report is generating."},
    {"file": "_chat", "kind": "router", "name": "_route_entry", "args": _STATE_ARG},
    {"file": "_chat", "kind": "tool", "name": "ask_user",
     "args": {"questions": ["Primary Research", "Due Diligence", "Standard Report"]}},
    {"file": "_chat", "kind": "helper", "name": "_extract_in_progress_layout",
     "args": {"messages": "$MESSAGES"}},
    {"file": "_chat", "kind": "helper", "name": "generate_chat_title",
     "args": {"user_query": "Give me a report on the UK fintech market",
              "ai_response": "Here is a proposed layout for that report."}},

    # ---- _retrieve.py ----
    {"file": "_retrieve", "kind": "tool", "name": "retrieve",
     "args": {"user_instructions": "Assess the UK fintech market.",
              "report_layout": "# UK Fintech\n\n## Market Size\n\n## Regulation\n",
              "report_language": "English", "report_title": "UK Fintech",
              "domain_name": "due_diligence", "report_type": "study"},
     "note": "domain_name='due_diligence' returns right after config is stored — "
             "use 'default' for the full context-gathering path."},
    {"file": "_retrieve", "kind": "tool", "name": "query_document",
     "args": {"user_query": "What does the document say about revenue?",
              "status_message": "Reading your document", "progress_updates": []},
     "note": "Needs an uploaded-document session; returns a notice otherwise."},
    {"file": "_retrieve", "kind": "tool", "name": "retrieve_latest_info",
     "args": {"user_query": "latest UK fintech funding rounds",
              "status_message": "Checking current sources", "progress_updates": []},
     "note": "Disabled in the graph, still callable directly."},
    {"file": "_retrieve", "kind": "node", "name": "_sequential_tools_node", "args": _STATE_ARG,
     "note": "Runs read-tool calls on the last AIMessage one at a time."},
    {"file": "_retrieve", "kind": "router", "name": "_route_after_tools", "args": _STATE_ARG},
    {"file": "_retrieve", "kind": "helper", "name": "_has_uploaded_documents", "args": {}},
    {"file": "_retrieve", "kind": "helper", "name": "_uses_grep_file_search", "args": {}},
    {"file": "_retrieve", "kind": "helper", "name": "_dispatch_read_tool",
     "args": {"name": "query_document", "args": {"user_query": "summarise the document"}}},

    # ---- _report.py ----
    {"file": "_report", "kind": "tool", "name": "propose_report_layout",
     "args": {"report_layout": "# UK Fintech\n\n## Market Size\n- Segments\n- Growth\n\n"
                               "## Regulation\n- FCA\n",
              "report_title": "UK Fintech"},
     "note": "Kicks off the background web refresh as a side effect."},
    {"file": "_report", "kind": "tool", "name": "update_proposed_report_layout",
     "args": {"report_layout": "# UK Fintech\n\n## Market Size\n\n## Regulation\n",
              "report_title": "UK Fintech"},
     "note": "The web refresh, run in the foreground. Slow — real search."},
    {"file": "_report", "kind": "helper", "name": "_web_refreshed_layout",
     "args": {"report_layout": "# UK Fintech\n\n## Market Size\n\n## Regulation\n",
              "report_title": "UK Fintech"}},
    {"file": "_report", "kind": "helper", "name": "_mint_sibling_layout",
     "args": {"report_layout": "# UK Fintech\n\n## Market Size\n\n## Regulation\n",
              "report_title": "UK Fintech", "target_tier": "brief"}},
    {"file": "_report", "kind": "helper", "name": "_parsed_layout_cards",
     "args": {"layout_markdown": "# UK Fintech\n\n## Market Size\n\n## Regulation\n"}},
    {"file": "_report", "kind": "node", "name": "layout_pair", "args": _STATE_ARG,
     "note": "Mints both tiers' layouts. Two real LLM passes."},
    {"file": "_report", "kind": "node", "name": "report_config", "args": _STATE_ARG,
     "needs_graph": True,
     "note": "Calls interrupt() — only works inside a graph run. Use flow mode."},
    {"file": "_report", "kind": "node", "name": "generate_report", "args": _STATE_ARG,
     "heavy": True,
     "note": "The real report build. Minutes, and real LLM spend per card."},
    {"file": "_report", "kind": "node", "name": "run_primary_research", "args": _STATE_ARG,
     "heavy": True, "note": "Primary-research subgraph. Needs an uploaded document."},
    {"file": "_report", "kind": "node", "name": "run_due_diligence", "args": _STATE_ARG,
     "heavy": True, "note": "Due-diligence subgraph. Real, slow."},
    {"file": "_report", "kind": "helper", "name": "_pending_retrieve_call",
     "args": {"messages": "$MESSAGES"}},
    {"file": "_report", "kind": "helper", "name": "_get_recent_tool_messages",
     "args": {"messages": "$MESSAGES"}},
    {"file": "_report", "kind": "helper", "name": "_patched_layout_tool_message",
     "args": {"messages": "$MESSAGES"}},

    # ---- _graph.py ----
    {"file": "_graph", "kind": "node", "name": "_gate_node", "args": _STATE_ARG,
     "note": "One-tool-per-step gate. Pure — no LLM call."},
    {"file": "_graph", "kind": "router", "name": "_route_after_report_or_respond",
     "args": _STATE_ARG},
    {"file": "_graph", "kind": "router", "name": "_route_after_gate", "args": _STATE_ARG},
    {"file": "_graph", "kind": "router", "name": "_route_by_domain", "args": _STATE_ARG},
    {"file": "_graph", "kind": "node", "name": "domain_router", "args": _STATE_ARG},
    {"file": "_graph", "kind": "helper", "name": "_count_turn_tool_calls",
     "args": {"messages": "$MESSAGES"}},
    {"file": "_graph", "kind": "helper", "name": "_layout_ever_proposed",
     "args": {"messages": "$MESSAGES"}},
    {"file": "_graph", "kind": "helper", "name": "graph_config", "args": {}},

    # ---- module-level functions ----
    {"file": "_helpers", "kind": "func", "name": "_normalize_report_config",
     "args": {"submission": {"report_tier": "brief", "style": "investor"},
              "default_tier": "study"}},
    {"file": "_helpers", "kind": "func", "name": "_sibling_tier", "args": {"tier": "study"}},
    {"file": "_helpers", "kind": "func", "name": "_keep_finished_chat_history",
     "args": {"messages": "$MESSAGE_DICTS"}},
    {"file": "_helpers", "kind": "func", "name": "_conversation_excerpt",
     "args": {"messages": "$MESSAGES", "max_chars": 6000}},
    {"file": "_helpers", "kind": "func", "name": "_proposed_layout_tool_result",
     "args": {"report_title": "UK Fintech",
              "report_layout": "# UK Fintech\n\n## Market Size\n",
              "web_updated": False}},
    {"file": "_helpers", "kind": "func", "name": "_normalize_tool_call",
     "args": {"call": {"name": "retrieve", "args": "{}", "id": "call_1"}}},
    {"file": "_helpers", "kind": "func", "name": "_topic_from_query",
     "args": {"query": "the latest UK fintech funding rounds", "max_words": 7}},
    {"file": "_helpers", "kind": "func", "name": "_heartbeat_messages",
     "args": {"search_query": "UK fintech", "progress_updates": []}},
]

_MODULES = {"_chat": _chat, "_retrieve": _retrieve, "_report": _report,
            "_graph": _graph, "_helpers": _helpers, "_state": _state}


def _resolve(target: dict, casper):
    """The callable for a registry entry — bound method, or module function."""
    if target["kind"] == "func":
        return getattr(_MODULES[target["file"]], target["name"])
    return getattr(casper, target["name"])


def build_catalog(casper) -> list[dict]:
    """Registry + live signature/docstring, so the page mirrors the real code."""
    out = []
    for entry in _REGISTRY:
        try:
            fn = _resolve(entry, casper)
        except AttributeError:
            continue
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(…)"
        doc = inspect.getdoc(fn) or ""
        out.append(
            {
                **entry,
                "signature": f"{entry['name']}{sig}",
                "doc": doc,
                "is_async": inspect.iscoroutinefunction(fn),
                "module": f"app.research.agent.{entry['file']}",
            }
        )
    return out


# ── the harness session ────────────────────────────────────────────────────
class HarnessSession:
    def __init__(self, casper, send):
        self.casper = casper
        self._send = send
        self.history: list = []
        self.layout_text: str = ""
        self._layout_watch: asyncio.Task | None = None
        self._emitted_updated_id: int | None = None
        # The graph parks here until a Confirm. A later chat message does not
        # resume it — only confirm_report_config does.
        self.paused_thread_id: str | None = None
        self.paused_payload: dict | None = None
        # `busy` gates FOREGROUND chat turns only. A report generating in the
        # background must not block chat — that is the whole point of
        # `respond_during_report`.
        self.busy = False
        self.report_task: asyncio.Task | None = None
        # Tracks the cards a report emits and turns a free-text "change this"
        # request into the refiner's `fe_json_for_refine` payload. One per
        # report; fed card events from the report stream and edit requests from
        # `respond_during_report`.
        self.refine_router: RefineRequestRouter | None = None
        self.refine_queue: list[dict] = []
        self._refine_pending: list[dict] = []
        self._refine_tasks: set[asyncio.Task] = set()

    # -- state ----------------------------------------------------------
    def set_report_in_progress(self, on: bool, title: str | None = None):
        self.casper.report_in_progress = on
        self.casper.in_progress_report_id = "harness-report-1" if on else None
        if title:
            self.casper.in_progress_report_title = title
        elif on and not getattr(self.casper, "in_progress_report_title", None):
            self.casper.in_progress_report_title = "Untitled report"

    def install_layout_hook(self):
        """Let the page decide what layout `respond_during_report` references."""
        harness = self

        def _extract(_messages):
            return harness.layout_text.strip()[:6000]

        self.casper._extract_in_progress_layout = _extract

    def state_dict(self) -> dict:
        c = self.casper
        return {
            "report_in_progress": bool(getattr(c, "report_in_progress", False)),
            "report_id": getattr(c, "in_progress_report_id", None),
            "report_title": getattr(c, "in_progress_report_title", None),
            "layout_chars": len(self.layout_text),
            "history_messages": len(self.history),
            "paused": bool(self.paused_thread_id),
            "paused_thread_id": self.paused_thread_id,
            "thread_id": getattr(c, "turn_thread_id", None),
            "has_documents": bool(self._safe_call(c._has_uploaded_documents)),
            "file_search_mode": getattr(c, "file_search_mode", None),
            "busy": self.busy,
            "report_running": bool(self.report_task and not self.report_task.done()),
            "refine_queued": len(self.refine_queue),
            "refine_tracking": self.refine_router is not None,
        }

    @staticmethod
    def _safe_call(fn):
        try:
            return fn()
        except Exception:
            return None

    async def emit(self, **payload):
        await self._send(payload)

    def _remember(self, final_messages) -> None:
        """Write the turn back into history without dropping earlier humans.

        A short ``values`` snapshot must not replace a longer conversation the
        page already has.
        """
        dumped: list = []
        if final_messages:
            dumped = [
                m.model_dump() if hasattr(m, "model_dump") else m
                for m in final_messages
                if getattr(m, "type", None) != "system"
                and not (isinstance(m, dict) and m.get("type") == "system")
            ]
            dumped = _helpers._keep_finished_chat_history(dumped)
        if len(dumped) >= len(self.history):
            self.history = dumped

    def _begin_fresh_turn(self) -> None:
        """Each ordinary turn gets its own checkpointer thread.

        The harness reuses one Casper for the whole session. Keeping the id
        minted at construction would append every turn onto the same
        checkpointed history and the messages reducer would duplicate.
        """
        self.casper.turn_thread_id = str(uuid7())
        self.casper._resuming_report_config = False
        self.casper.report_config_submission = {}

    # -- flow mode ------------------------------------------------------
    async def send_turn(self, text: str):
        await self.emit(type="turn_start", text=text)
        await self.emit(type="chat", role="user", text=text)

        # A chat message while paused does not resume the parked thread. It
        # starts a new turn; the old pause stays in the checkpointer but is no
        # longer the one this session will confirm.
        self.paused_thread_id = None
        self.paused_payload = None
        self._begin_fresh_turn()

        self.casper.user_previous_messages = list(self.history)
        gs = await self.casper.get_processing_state(text)

        if isinstance(gs, dict):  # error path
            await self.emit(type="error", message=gs.get("error", "get_processing_state failed"))
            await self.emit(type="turn_end", handled_by="?", path=[], edit_events=0)
            return

        await self._drain_stream(gs, persist_history=True)

    async def confirm_report_config(self, submission: dict):
        """Resume the parked turn. The only thing that does."""
        if not self.paused_thread_id:
            await self.emit(
                type="error", message="This chat is not waiting for a report configuration."
            )
            return

        self.casper.turn_thread_id = self.paused_thread_id
        await self.emit(type="turn_start", text="(confirm report config)")
        await self.emit(
            type="chat",
            role="user",
            text=f"Confirm · {submission.get('report_tier', '?')} · "
                 f"{submission.get('style', 'investor')}",
        )

        gs = self.casper.resume_report_config(submission)
        self.paused_thread_id = None
        self.paused_payload = None

        # Report generation runs for minutes. Draining it inline blocked the
        # socket's receive loop for that whole time, so a follow-up message could
        # not even be READ, let alone answered. Run it in the background and flag
        # the chat as report-in-progress — which is what routes follow-ups to
        # `respond_during_report`, the node that exists for exactly this.
        title = (self.casper.retrieve_config or {}).get("report_title") or "Report"
        self.set_report_in_progress(True, title)
        # Fresh tracker per report — must exist before the first card streams.
        self.refine_router = None
        self.refine_queue = []
        self._ensure_refine_router()
        self.report_task = asyncio.create_task(self._run_report_in_background(gs))
        await self.emit(type="state", **self.state_dict())

    async def _run_report_in_background(self, gs):
        try:
            # persist_history=False: the chat transcript belongs to the chat turns.
            # The report's own graph messages must not clobber it, and a concurrent
            # `respond_during_report` turn is writing to the same history.
            await self._drain_stream(gs, persist_history=False, resume=True)
        except Exception as exc:
            await self.emit(
                type="error", message=f"report generation failed: {exc!r}",
                trace=traceback.format_exc(),
            )
        finally:
            self.set_report_in_progress(False)
            await self.emit(type="report_done")
            await self.emit(type="state", **self.state_dict())

    async def _drain_stream(self, gs, *, persist_history: bool, resume: bool = False):
        started = time.time()
        nodes_seen: list[str] = []
        edit_events = 0
        streamed_any_text = False
        final_messages = None
        cur_stream_node = None
        saw_propose = False
        emitted_updated = False
        paused_for_report_config = False
        pause_payload: dict | None = None

        active_node = None
        last_model_activity = 0.0
        async for mode, output in gs:
            if mode == "messages":
                chunk, meta = output
                node = (meta or {}).get("langgraph_node", "")
                is_ai = getattr(chunk, "type", "") == "AIMessageChunk"
                piece = _chunk_text(chunk)
                # `updates` only lands when a node RETURNS, so a node that is
                # still working (report_or_respond waiting out the layout
                # refresh) would otherwise never show as the current one.
                if node and node != active_node:
                    active_node = node
                    await self.emit(
                        type="node_active", node=node,
                        elapsed=round(time.time() - started, 2),
                    )
                await self.emit(
                    type="raw", mode="messages",
                    data={"node": node, "msg_type": getattr(chunk, "type", ""),
                          "text": piece, "name": getattr(chunk, "name", None)},
                )
                if node in ("report_or_respond", "respond_during_report") and is_ai and piece:
                    if cur_stream_node != node:
                        cur_stream_node = node
                        await self.emit(type="assistant_start", node=node)
                    await self.emit(type="assistant_token", node=node, text=piece)
                    streamed_any_text = True

            elif mode == "updates":
                # A node returned, so the next streaming burst is a NEW assistant
                # message. Without this, a preamble written before a tool call and
                # the reply written after it merge into one bubble, with the tool's
                # output rendered somewhere else entirely.
                cur_stream_node = None
                for node_name, payload in (output or {}).items():
                    nodes_seen.append(node_name)
                    await self.emit(
                        type="raw", mode="updates",
                        data={"node": node_name, "payload": _safe(_dump_msgs(payload))},
                    )
                    await self.emit(
                        type="node", node=node_name, elapsed=round(time.time() - started, 2)
                    )
                    if node_name == "__interrupt__":
                        paused_for_report_config = True

            elif mode == "custom":
                evt = _safe(output) if isinstance(output, dict) else {"value": _safe(output)}
                name = evt.get("name", "")
                status = evt.get("status", "")
                # When the model stopped talking. Anything after this is the
                # graph waiting on something (usually the layout refresh), which
                # is what makes a turn far longer than the LLM work in it.
                if status == "message_stream_complete":
                    last_model_activity = time.time() - started
                await self.emit(type="raw", mode="custom", data=evt)
                await self.emit(type="event", name=name, status=status, data=evt)

                if name == "post_report_edit_request":
                    edit_events += 1
                    await self.emit(type="edit_request", data=evt)
                    # Hand it to the router: it picks the section/subsection the
                    # user meant and produces the refiner's fe_json payload.
                    self._spawn_refine_routing(evt)
                elif name == "layout_pair":
                    await self.emit(
                        type="layout_option", report_tier=evt.get("report_tier", ""),
                        title=evt.get("report_title", ""),
                        cards=evt.get("report_layout", []), markdown=evt.get("markdown", ""),
                    )
                elif name == "report_config" and status == "awaiting_configuration":
                    paused_for_report_config = True
                    pause_payload = evt
                elif name in ("propose_report_layout", "updated_proposed_report_layout") \
                        and "report_layout" in evt:
                    is_upd = name == "updated_proposed_report_layout"
                    saw_propose = saw_propose or not is_upd
                    emitted_updated = emitted_updated or is_upd
                    if is_upd:
                        _u = getattr(self.casper, "updated_proposed_report_layout", None)
                        if _u is not None:
                            self._emitted_updated_id = id(_u)
                    await self.emit(
                        type="layout_proposal", updated=is_upd,
                        title=evt.get("report_title", ""),
                        change_summary=evt.get("change_summary", ""),
                        cards=evt.get("report_layout", []),
                    )
                    # `_watch_layout_refresh` only reports "done" for a refresh
                    # that outlives the turn. When the update arrives inside the
                    # turn's own stream (report_or_respond awaited it), close the
                    # refresh out here or the page keeps saying "refreshing…".
                    await self.emit(
                        type="layout_refresh", status="done" if is_upd else "running"
                    )
                elif name == "ask_user" and status == "options":
                    await self.emit(type="ask_user", questions=evt.get("questions", []) or [])
                elif name == "retrieve" and "report_layout" in evt:
                    await self.emit(
                        type="final_layout", markdown=evt.get("report_layout", ""),
                        report_type=evt.get("report_type", ""),
                    )
                elif name in ("generate_report", "primary_research", "due_diligence"):
                    # Real report generation — surface its progress as report output.
                    await self.emit(type="report_progress", status=status, data=evt)
                    if evt.get("card_db"):
                        await self._ingest_card(evt)

            elif mode == "values":
                msgs = (output or {}).get("messages") or []
                # Keep the longest snapshot. A later node can emit an empty
                # `messages` update; overwriting here is how the topic vanished
                # after ask_user.
                if msgs and (final_messages is None or len(msgs) >= len(final_messages)):
                    final_messages = msgs
                await self.emit(
                    type="raw", mode="values",
                    data={"count": len(msgs), "messages": [_msg_digest(m) for m in msgs]},
                )

        if paused_for_report_config:
            payload = pause_payload or {
                "thread_id": self.casper.turn_thread_id, "default_tier": "study",
                "tiers": ["study", "brief"], "styles": ["investor"], "layouts": [],
            }
            self.paused_thread_id = payload.get("thread_id") or self.casper.turn_thread_id
            self.paused_payload = payload
            # A paused turn still happened. Without this the pause path returns
            # before `_remember`, so if the user types a chat message instead of
            # confirming, the next turn is rebuilt from a history that never saw
            # this exchange — the conversation appears to lose its topic.
            if persist_history:
                self._remember(final_messages)
            await self.emit(type="report_config_required", **payload)
            _total = time.time() - started
            await self.emit(
                type="turn_end", handled_by="report_config", path=nodes_seen,
                edit_events=edit_events, paused=True,
                elapsed=round(_total, 2),
                model_seconds=round(last_model_activity, 2),
                waiting_seconds=round(max(0.0, _total - last_model_activity), 2),
            )
            await self.emit(type="state", **self.state_dict())
            return

        # fallback: surface a non-streamed final AI message
        if not streamed_any_text and final_messages:
            last_ai = next(
                (m for m in reversed(final_messages) if getattr(m, "type", None) == "ai"), None
            )
            body = _chunk_text(last_ai) if last_ai is not None else ""
            if body.strip():
                await self.emit(
                    type="chat", role="assistant", text=body, node=cur_stream_node or ""
                )

        entry = next(
            (n for n in nodes_seen if n in ("respond_during_report", "report_or_respond")), "?"
        )
        if resume and entry == "?":
            entry = "report_config"
        _total = time.time() - started
        await self.emit(
            type="turn_end", handled_by=entry, path=nodes_seen, edit_events=edit_events,
            elapsed=round(_total, 2),
            model_seconds=round(last_model_activity, 2),
            waiting_seconds=round(max(0.0, _total - last_model_activity), 2),
        )

        if persist_history:
            self._remember(final_messages)
        await self.emit(type="state", **self.state_dict())

        # The web refresh of a proposed layout runs as a background task. The
        # graph only waits it out on a turn that ends WITHOUT a tool call, so on
        # an ask_user / retrieve turn `updated_proposed_report_layout` never
        # reaches this turn's stream. Watch the task and push it when it lands.
        task = getattr(self.casper, "_layout_refresh_task", None)
        already_have = getattr(self.casper, "updated_proposed_report_layout", None)
        if not emitted_updated:
            if (
                already_have is not None
                and id(already_have) != self._emitted_updated_id
                and (task is None or task.done())
            ):
                await self._emit_updated_layout()
            elif saw_propose and task is not None and not task.done():
                if self._layout_watch and not self._layout_watch.done():
                    self._layout_watch.cancel()
                self._layout_watch = asyncio.create_task(self._watch_layout_refresh(task))

    # -- refine request routing -----------------------------------------
    def _ensure_refine_router(self) -> RefineRequestRouter:
        """One router per report. Its own writer is a graph-run stream writer,
        which does not resolve here — we call it outside any graph — so it is
        handed a plain collector the session drains."""
        if self.refine_router is None:
            self.refine_router = RefineRequestRouter(
                chat_id=self.casper.chat_id,
                report_id=getattr(self.casper, "in_progress_report_id", None),
                report_title=getattr(self.casper, "in_progress_report_title", "") or "",
                user_id=getattr(self.casper, "user_id", None),
                event_writer=self._refine_pending.append,
            )
        return self.refine_router

    async def _ingest_card(self, evt: dict) -> None:
        """Feed one generate_report card frame to the tracker.

        Cheap: a local MemorySaver graph step, no LLM. `parse_card_event`
        ignores status/heartbeat frames itself.
        """
        try:
            await self._ensure_refine_router().ingest_card_event(evt)
        except Exception as exc:
            # Surface it: a silently untracked card means a later edit request
            # cannot be routed to that section, which is exactly the kind of
            # thing this harness exists to make visible.
            await self.emit(
                type="refine_error",
                message=f"card ingest failed: {exc!r}",
                section=((evt.get("card_db") or {}).get("section") or [{}])[0].get("name"),
            )

    def _spawn_refine_routing(self, evt: dict) -> None:
        """Route a captured edit request without stalling the stream it came from.

        Routing costs one LLM call, so it must not block the report drain or the
        chat turn that produced the request.
        """
        task = asyncio.create_task(self._route_edit_request(evt))
        self._refine_tasks.add(task)
        task.add_done_callback(self._refine_tasks.discard)

    async def _route_edit_request(self, evt: dict) -> None:
        router = self._ensure_refine_router()
        await self.emit(type="refine_routing", request=evt.get("request"))
        try:
            await router.handle_post_report_edit_request(evt)
        except Exception as exc:
            await self.emit(
                type="refine_error", message=f"routing failed: {exc!r}",
                trace=traceback.format_exc(),
            )
            return
        await self._flush_refine_events()

    async def _flush_refine_events(self) -> None:
        """Forward whatever the router wrote, and queue each routed request."""
        while self._refine_pending:
            raw = _safe(self._refine_pending.pop(0))
            if not isinstance(raw, dict):
                continue
            await self.emit(type="raw", mode="custom", data=raw)
            if raw.get("name") != ROUTED_EVENT_NAME:
                continue
            # Shape emitted by RefineRequestRouter: raw_user_message + a `target`
            # dict (kind / section_id / subsection_id / matched_name / confidence
            # / reasoning) + the fe_json_for_refine payload the refiner consumes.
            target = raw.get("target") or {}
            entry = {
                "request": raw.get("raw_user_message"),
                "status": raw.get("status"),
                "target_kind": target.get("kind"),
                "matched_name": target.get("matched_name"),
                "section_id": target.get("section_id"),
                "subsection_id": target.get("subsection_id"),
                "confidence": target.get("confidence"),
                "reasoning": target.get("reasoning"),
                "fe_json_for_refine": raw.get("fe_json_for_refine"),
                "at": time.strftime("%H:%M:%S"),
            }
            self.refine_queue.append(entry)
            await self.emit(
                type="refine_request", entry=entry, queue_len=len(self.refine_queue)
            )
            await self.emit(type="state", **self.state_dict())

    def _cards_from_updated(self, updated) -> list:
        """Card list for a ``updated_proposed_report_layout`` object.

        Reuses the app's own converters (they live in ``_report`` since the
        model split) so the shape matches ``propose_report_layout`` exactly.
        """
        try:
            md = _report._updated_layout_to_markdown(updated)
            cleaned = _report.parse_markdown_report_layout(md)
            return _report.modify_report_layout(cleaned) or []
        except Exception:
            return []

    async def _emit_updated_layout(self):
        updated = getattr(self.casper, "updated_proposed_report_layout", None)
        if updated is None:
            return
        self._emitted_updated_id = id(updated)
        await self.emit(
            type="layout_proposal", updated=True,
            title=getattr(updated, "title", "") or "",
            change_summary=(getattr(updated, "change_summary", "") or "").strip(),
            cards=self._cards_from_updated(updated),
        )

    async def _watch_layout_refresh(self, task: asyncio.Task):
        try:
            await self.emit(type="layout_refresh", status="running")
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=200)
            except asyncio.TimeoutError:
                await self.emit(type="layout_refresh", status="timeout")
                return
            except asyncio.CancelledError:
                await self.emit(type="layout_refresh", status="superseded")
                return
            except Exception as exc:
                await self.emit(type="layout_refresh", status="error", detail=repr(exc))
                return
            if getattr(self.casper, "updated_proposed_report_layout", None) is not None:
                await self._emit_updated_layout()
                await self.emit(type="layout_refresh", status="done")
            else:
                await self.emit(type="layout_refresh", status="no_change")
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # -- node mode ------------------------------------------------------
    async def run_node(self, name: str, raw_args: dict, use_system: bool):
        """Invoke one target directly and report everything it did."""
        target = next((t for t in _REGISTRY if t["name"] == name), None)
        if target is None:
            await self.emit(type="node_error", name=name, message=f"unknown target {name!r}")
            return
        if target.get("needs_graph"):
            await self.emit(
                type="node_error", name=name,
                message=f"{name} calls interrupt() and only works inside a graph run — "
                        f"drive it from flow mode instead.",
            )
            return

        fn = _resolve(target, self.casper)
        system = self.casper.system_message if use_system else None
        try:
            kwargs = self._materialise(raw_args, system)
        except Exception as exc:
            await self.emit(type="node_error", name=name, message=f"bad inputs: {exc}",
                            trace=traceback.format_exc())
            return

        await self.emit(type="node_start", name=name, kind=target["kind"],
                        file=target["file"], args=_safe(raw_args))

        events: list = []
        pending: list = []

        def sink(evt):
            # Called synchronously from inside the node, possibly off the loop
            # thread (card work runs in executors), so this only appends — a
            # drainer on the loop does the awaiting.
            payload = _safe(evt)
            events.append(payload)
            pending.append(payload)

        async def drain():
            while True:
                while pending:
                    await self.emit(type="node_event", name=name, data=pending.pop(0))
                await asyncio.sleep(0.05)

        started = time.time()
        drainer = asyncio.create_task(drain())
        try:
            with capture_stream_events(sink):
                result = fn(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
        except Exception as exc:
            drainer.cancel()
            while pending:
                await self.emit(type="node_event", name=name, data=pending.pop(0))
            await self.emit(
                type="node_result", name=name, ok=False, error=repr(exc),
                trace=traceback.format_exc(), elapsed=round(time.time() - started, 2),
                events=events,
            )
            return
        finally:
            drainer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drainer

        while pending:  # flush anything the drainer missed on its last tick
            await self.emit(type="node_event", name=name, data=pending.pop(0))

        await self.emit(
            type="node_result", name=name, ok=True,
            result=_safe(_dump_msgs(result) if isinstance(result, dict) else result),
            result_type=type(result).__name__,
            elapsed=round(time.time() - started, 2), events=events,
        )
        await self.emit(type="state", **self.state_dict())

    def _materialise(self, raw_args: dict, system: SystemMessage | None) -> dict:
        """Expand the page's JSON into real call arguments.

        ``"$MESSAGES"`` becomes LangChain message objects, ``"$MESSAGE_DICTS"``
        their ``model_dump()`` form (what ``_keep_finished_chat_history`` takes),
        and ``"$HISTORY"`` splices in the live flow-mode conversation.
        """
        def expand(value):
            if value == "$HISTORY":
                return _mk_messages(
                    [m for m in self.history if isinstance(m, dict)], system
                )
            if isinstance(value, list) and value and value[0] == "$MESSAGE_DICTS":
                msgs = _mk_messages(value[1:], system)
                return [m.model_dump() for m in msgs]
            if isinstance(value, list):
                return [expand(v) for v in value]
            if isinstance(value, dict):
                return {k: expand(v) for k, v in value.items()}
            return value

        out = {}
        for key, value in (raw_args or {}).items():
            if key == "state":
                specs = (value or {}).get("messages")
                if specs == "$HISTORY":
                    msgs = _mk_messages([m for m in self.history if isinstance(m, dict)], system)
                else:
                    msgs = _mk_messages(specs, system)
                out["state"] = {**{k: v for k, v in (value or {}).items() if k != "messages"},
                                "messages": msgs}
            elif key == "messages":
                if value == "$HISTORY":
                    out["messages"] = _mk_messages(
                        [m for m in self.history if isinstance(m, dict)], system
                    )
                elif isinstance(value, list) and value and value[0] == "$MESSAGE_DICTS":
                    out["messages"] = [m.model_dump() for m in _mk_messages(value[1:], system)]
                else:
                    out["messages"] = _mk_messages(value, system)
            else:
                out[key] = expand(value)
        return out


# ── build the real Casper (nothing stubbed) ────────────────────────────────
async def build_casper():
    cfg = {
        "user_name": "harness",
        "chat_id": "harness-chat-1",
        "user_previous_messages": [],
        "user_id": "harness-user",
    }
    casper = Casper(cfg)
    await casper.async_init()
    return casper


SAMPLE_LAYOUT = """# EV Market Outlook 2026

## Executive Summary
## Market Size and Growth
## Competitive Landscape
## Battery Technology
## Policy and Regulation
## Risks
## Outlook
"""


# ── websocket ──────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    asyncio.get_running_loop().set_exception_handler(_swallow_infra_noise)
    send_lock = asyncio.Lock()

    async def send(payload: dict):
        async with send_lock:
            await ws.send_text(json.dumps(_safe(payload), default=str))

    await send({"type": "status", "text": "building casper (real LLM clients, real report nodes)…"})
    try:
        casper = await build_casper()
    except Exception as exc:  # pragma: no cover
        await send({"type": "error", "message": f"build_casper failed: {exc!r}",
                    "trace": traceback.format_exc()})
        await ws.close()
        return

    h = HarnessSession(casper, send)
    h.install_layout_hook()
    h.set_report_in_progress(False)
    await send({"type": "ready", "sample_layout": SAMPLE_LAYOUT,
                "catalog": build_catalog(casper), "caspr_core": str(_CASPR_CORE)})
    await send({"type": "state", **h.state_dict()})

    async def guard(coro):
        """Run one command, keeping the busy flag honest even if it raises."""
        h.busy = True
        await send({"type": "state", **h.state_dict()})
        try:
            await coro
        finally:
            h.busy = False
            await send({"type": "state", **h.state_dict()})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except ValueError:
                await send({"type": "error", "message": "bad json"})
                continue

            kind = msg.get("type")
            try:
                if kind == "send":
                    text = (msg.get("text") or "").strip()
                    if text:
                        await guard(h.send_turn(text))
                elif kind == "run_node":
                    await guard(
                        h.run_node(
                            msg.get("name") or "",
                            msg.get("args") or {},
                            bool(msg.get("use_system", True)),
                        )
                    )
                elif kind == "confirm_report_config":
                    await (
                        h.confirm_report_config(
                            {
                                "report_tier": (msg.get("report_tier") or "study").strip().lower(),
                                "style": (msg.get("style") or "investor").strip().lower(),
                                "output_formats": msg.get("output_formats") or [],
                                "language": msg.get("language") or "English",
                                "data_sources": msg.get("data_sources") or [],
                            }
                        )
                    )
                elif kind == "report_on":
                    h.set_report_in_progress(True, (msg.get("title") or "").strip() or None)
                    await send({"type": "state", **h.state_dict()})
                elif kind == "report_off":
                    h.set_report_in_progress(False)
                    await send({"type": "state", **h.state_dict()})
                elif kind == "layout":
                    h.layout_text = msg.get("text") or ""
                    await send({"type": "state", **h.state_dict()})
                elif kind == "scenario":
                    h.layout_text = SAMPLE_LAYOUT
                    h.set_report_in_progress(True, "EV Market Outlook 2026")
                    h.history = []
                    await send({"type": "scenario_ready", "layout": SAMPLE_LAYOUT})
                    await send({"type": "state", **h.state_dict()})
                elif kind == "reset":
                    h.history = []
                    h.paused_thread_id = None
                    h.paused_payload = None
                    h._begin_fresh_turn()
                    await send({"type": "reset_done"})
                    await send({"type": "state", **h.state_dict()})
                elif kind == "state":
                    await send({"type": "state", **h.state_dict()})
                elif kind == "history":
                    await send({"type": "history",
                                "messages": [_msg_digest(m) for m in h.history]})
                elif kind == "refine_queue":
                    await send({"type": "refine_queue", "queue": h.refine_queue})
                elif kind == "refine_cards":
                    cards = await h._ensure_refine_router().tracked_cards()
                    await send({"type": "refine_cards", "cards": _safe(cards)})
                elif kind == "catalog":
                    await send({"type": "catalog", "catalog": build_catalog(casper)})
                else:
                    await send({"type": "error", "message": f"unknown command {kind!r}"})
            except Exception as exc:
                await send({"type": "error", "message": f"command failed: {exc!r}",
                            "trace": traceback.format_exc()})
    except WebSocketDisconnect:
        pass
    except Exception:
        with contextlib.suppress(Exception):
            await send({"type": "error", "message": traceback.format_exc()})


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HARNESS_HOST", "127.0.0.1")
    port = int(os.environ.get("HARNESS_PORT", "8765"))
    print(f"\n  caspr agent harness → http://{host}:{port}")
    print(f"  caspr core          → {_CASPR_CORE}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
