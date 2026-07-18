#!/usr/bin/env python3
"""ALFWorld SkillDAG runner — dispatched from ``alfworld_run.py --mode skilldag``.

Wires the ALFWorld play loop to the SkillDAG CLI (same CLI surface as the
SkillsBench integration, host-side here):

    1. Bootstrap:    ``skilldag initialize-graph`` (run once, if graph missing).
    2. Retrieve:     Agent emits ``{"command": "skilldag graph search ..."}``
                     which calls the host CLI and returns stdout.
    3. Online edit:  Agent emits ``{"command": "skilldag graph edit-edge
                     ..."}``. Mutations apply immediately with no
                     cross-process lock — experiment concurrency is low enough
                     that rare lost updates are acceptable.

LLM path: litellm with ``openai/gpt-5.2-codex`` as the primary default (for
the open-source reproducibility path). Set ``SKILLDAG_BACKBONE=openrouter``
to use ``minimax/minimax-m2.7`` via OpenRouter (optional, requires API key).
``add_anthropic_caching`` helper is inlined to mark cache breakpoints.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml
from retry import retry


# --- path setup ------------------------------------------------------------
# Locate this SkillDAG checkout; override via env for other layouts.
SKILLDAG_HOME = Path(
    os.environ.get(
        "SKILLDAG_HOME",
        str(Path(__file__).resolve().parents[2]),
    )
).resolve()
SKILLDAG_SRC = SKILLDAG_HOME / "src"
for _path in (SKILLDAG_SRC, SKILLDAG_HOME):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import litellm  # noqa: E402

from skilldag.graph import SkillGraph  # noqa: E402  — pip-installed `skilldag` package (lowercase per PEP 8)
from benchmarks.shared.skilldag_prompt import (  # noqa: E402
    cli_reference,
    failure_reflection_protocol,
    when_to_mutate,
    mutation_rules,
    few_shot,
    search_protocol,
    success_path_discovery,
)
from benchmarks.skill_use_metrics import (  # noqa: E402
    attach_alfworld_skill_use_metric,
    print_summary as print_skill_use_summary,
    write_alfworld_skill_use_metrics,
)

def add_anthropic_caching(messages, model_name):  # type: ignore
    """Inline copy of terminus_2's anthropic_caching.add_anthropic_caching.

    Marks the last 2 messages' content as ``ephemeral`` cache breakpoints for
    Anthropic models. No-op for non-anthropic/claude models.
    """
    if not isinstance(model_name, str):
        return messages
    lower = model_name.lower()
    if "anthropic" not in lower and "claude" not in lower:
        return messages
    if not messages:
        return messages

    out = list(messages)
    # convert last 2 text-content messages to block-content with cache_control
    breakpoints_left = 2
    for idx in range(len(out) - 1, -1, -1):
        if breakpoints_left <= 0:
            break
        msg = dict(out[idx])
        content = msg.get("content")
        if isinstance(content, str) and content:
            msg["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
            out[idx] = msg
            breakpoints_left -= 1
        elif isinstance(content, list) and content:
            # already block form — attach cache_control to last text block
            blocks = [dict(b) for b in content]
            for b in reversed(blocks):
                if isinstance(b, dict) and b.get("type") == "text":
                    b["cache_control"] = {"type": "ephemeral"}
                    break
            msg["content"] = blocks
            out[idx] = msg
            breakpoints_left -= 1
    return out


def is_openrouter_chat_model(model_name: str, api_base: str | None = None) -> bool:
    lower_model = (model_name or "").lower()
    lower_base = (api_base or "").lower()
    return lower_model.startswith("openrouter/") or "openrouter.ai" in lower_base


def openrouter_payload_model(model_name: str) -> str:
    """Map local OpenRouter aliases to the model id sent to OpenRouter."""

    model_name = model_name.strip()
    aliases = {
        "openrouter/minimax-m2.7": "minimax/minimax-m2.7",
        "openrouter/minimax/minimax-m2.7": "minimax/minimax-m2.7",
    }
    lower = model_name.lower()
    if lower in aliases:
        return aliases[lower]
    if lower.startswith("openrouter/"):
        return model_name.split("/", 1)[1]
    return model_name


def normalize_openrouter_usage(usage_obj: dict) -> dict:
    """Normalize OpenRouter usage to local token buckets.

    OpenRouter reports prompt caching under ``prompt_tokens_details``. We store
    uncached prompt/cache-read/cache-write separately so downstream totals can
    sum ``prompt + completion + cache_read + cache_create`` without double
    counting the prompt total.
    """

    details = usage_obj.get("prompt_tokens_details") or {}
    prompt_total = int(usage_obj.get("prompt_tokens", 0) or 0)
    completion = int(usage_obj.get("completion_tokens", 0) or 0)
    cached = int(
        details.get("cached_tokens")
        or details.get("cache_read_tokens")
        or usage_obj.get("cache_read_input_tokens", 0)
        or 0
    )
    cache_write = int(
        details.get("cache_write_tokens")
        or details.get("cache_creation_tokens")
        or usage_obj.get("cache_creation_input_tokens", 0)
        or 0
    )
    return {
        "prompt_tokens": max(prompt_total - cached - cache_write, 0),
        "completion_tokens": completion,
        "cache_read": cached,
        "cache_create": cache_write,
        "prompt_tokens_total": prompt_total,
    }


def extract_chat_completion_content(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)

import alfworld  # noqa: E402,F401
from alfworld.agents.environment import get_environment  # noqa: E402


# --- constants -------------------------------------------------------------

# ALFWorld data dir: must be set via ``ALFWORLD_DATA`` env (alfworld's own
# convention). Falls back to a path relative to this checkout for convenience
# but OSS users typically export ``ALFWORLD_DATA`` after ``alfworld-download``.
DEFAULT_ALFWORLD_DATA = os.environ.get(
    "ALFWORLD_DATA", str(SKILLDAG_HOME / "data/alfworld/data")
)
DEFAULT_BASE_CONFIG = str(
    Path(__file__).resolve().parent / "base_config.yaml"
)
# Retrieval config: aligned with the bundled ALFWorld skill pool and graph.
DEFAULT_SKILLS_DIR = "data/alfworld_skills"
DEFAULT_GRAPH_PATH = "data/skilldag_graphs/skillgraph_alfworld.json"
DEFAULT_MODEL = "openai/gpt-5.2-codex"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

LLM_REQUEST_TIMEOUT_SECS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECS", "90"))


# --- util ------------------------------------------------------------------

class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def process_ob(ob: str) -> str:
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


def sanitize_response(response: str) -> str:
    """Trim common reasoning leakage before structured parsing."""
    if not response:
        return ""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    return response.strip()


def _extract_skill_id(command: str) -> str:
    """Best-effort: if the CLI command references a skill id (as a positional
    arg to ``skilldag show`` or ``skilldag graph <sub> <id>``), return that
    first id. Used only for audit (``loaded_skills`` in result JSON).
    """
    try:
        toks = shlex.split(command)
    except ValueError:
        return ""
    if len(toks) < 2 or toks[0] != "skilldag":
        return ""
    # Top-level: skilldag show <id>
    if toks[1] == "show" and len(toks) >= 3:
        return toks[2]
    # Graph subcommand: skilldag graph <sub> <id> ...
    if toks[1] == "graph" and len(toks) >= 4:
        sub = toks[2]
        if sub in {"show", "get-skill", "get-dependencies", "get-alternatives",
                  "get-conflicts", "check-set", "expand-set", "repair-set"}:
            return toks[3]
        if sub == "edit-edge" and len(toks) >= 5:
            return toks[4]
    return ""


def _extract_first_json_object(response: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            payload, _ = decoder.raw_decode(response[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def parse_turn_payload(response: str) -> tuple[dict | None, str | None]:
    """Parse one structured agent turn.

    Expected format:
      {"thought": "...", "action": "..."}
    or
      {"thought": "...", "command": "..."}
    """
    payload = _extract_first_json_object(response)
    if payload is None:
        return None, (
            "Your reply must be exactly one JSON object with `thought` and "
            "exactly one of `action` or `command`."
        )

    thought = payload.get("thought")
    action = payload.get("action")
    command = payload.get("command")

    if not isinstance(thought, str) or not thought.strip():
        return None, "Field `thought` must be a non-empty string."

    action_present = isinstance(action, str) and bool(action.strip())
    command_present = isinstance(command, str) and bool(command.strip())

    if action_present == command_present:
        return None, "Provide exactly one of `action` or `command`."

    if action_present:
        action = re.sub(
            r"^put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)$",
            r"move \1 to \2",
            action.strip(),
            flags=re.IGNORECASE,
        )

    return {
        "thought": thought.strip(),
        "action": action if action_present else "",
        "command": command.strip() if command_present else "",
    }, None


ALFWORLD_SYSTEM_PROMPT = f"""You are an ALFWorld agent. On each turn, output EXACTLY one JSON object:

(A) household action:
    {{"thought": "<one short sentence>", "action": "<env action>"}}

(B) shell command:
    {{"thought": "<one short sentence>", "command": "<single shell command>"}}

Valid household actions (A):
  go to {{recep}}
  take {{obj}} from {{recep}}
  move {{obj}} to {{recep}}
  open {{recep}} | close {{recep}}
  use {{obj}}
  clean {{obj}} with {{recep}}
  heat {{obj}} with {{recep}}
  cool {{obj}} with {{recep}}

The `command` field runs through the same generic shell-command path as any
other CLI call. Commands execute from the SkillDAG repository root. The
`skilldag` executable is already on PATH, so call it like any normal command.

{cli_reference()}

Output format rules:
- Output EXACTLY one JSON object and nothing else.
- Use exactly one of `action` or `command`, never both.
- Do NOT write an Observation field — that comes back on the next turn.
- Do NOT use <think>...</think>; put reasoning in the `thought` field.
- `command` must be a single-line shell command.

Strategy hints:
- At task start you have NO skill bodies loaded. First call `skilldag graph search "..."`; then `skilldag show <id>` for the skills that look relevant; then act.
- You may also run simple shell inspection commands such as `which skilldag` or `pwd` when needed, but be purposeful.
- Each command turn counts against your turn budget but NOT against your env-step budget.
- Treat task failure as evidence, not noise. After a failed action, first explain why it failed before acting again.
- If the same failure pattern repeats, do NOT keep paraphrasing the search query. Either test a materially different world-state hypothesis or emit a graph mutation backed by concrete evidence.
- Only mutate the graph when the failure indicates a reusable relation between two skills, not a one-off room/state mistake.
- Each task is fresh — do NOT claim a prior task is complete; always work on the current goal."""


def _alfworld_system_prompt() -> str:
    return ALFWORLD_SYSTEM_PROMPT


def _skilldag_task_header(task_id: str, runtime: "SkillDAGRuntime") -> str:
    return f"""SkillDAG graph: {runtime.graph_path}
Skills directory: {runtime.skills_dir}
No skill bodies are pre-loaded. Use `{{"command": "skilldag graph search \\"...\\" --top-k 5"}}`
to find skills, then `{{"command": "skilldag show <id>"}}` to read them.

{search_protocol()}

{failure_reflection_protocol()}

{success_path_discovery()}

{when_to_mutate()}

{mutation_rules()}

{few_shot()}
"""


# --- SkillDAG runtime ------------------------------------------------------

class SkillDAGRuntime:
    """Per-task wrapper around SkillGraph for retrieve + online edit + reflect."""

    def __init__(self, graph_path: str, skills_dir: str, *, task_id: str | None = None):
        self.graph_path = Path(graph_path)
        self.skills_dir = Path(skills_dir)
        self.task_id = task_id
        # Ensure graph exists
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"SkillDAG graph not found at {self.graph_path}. "
                "Run initialize-graph first (or pass --auto_bootstrap)."
            )
        self.graph = SkillGraph.load(
            graph_path=self.graph_path, skills_dir=self.skills_dir
        )

    def read_skill_md(self, skill_id: str) -> str:
        """Return SKILL.md body; empty if absent."""
        md = self.skills_dir / skill_id / "SKILL.md"
        return md.read_text(encoding="utf-8", errors="ignore") if md.exists() else ""


# --- shell command invocation ----------------------------------------------

COMMAND_TIMEOUT_SECS = float(os.environ.get("ALFWORLD_COMMAND_TIMEOUT_SECS", "60"))
COMMAND_OUTPUT_MAX_CHARS = int(os.environ.get("ALFWORLD_COMMAND_OUTPUT_MAX_CHARS", "3500"))
_MUTATION_SUBCMDS = {"edit-edge"}


def _is_mutation_command(tokens: list[str]) -> bool:
    """Return True if ``skilldag graph edit-edge ...`` is being invoked."""
    try:
        idx_graph = tokens.index("graph")
    except ValueError:
        return False
    sub = tokens[idx_graph + 1] if idx_graph + 1 < len(tokens) else ""
    return sub in _MUTATION_SUBCMDS


def _ensure_skilldag_wrapper() -> Path:
    """Expose `skilldag` as a normal command on PATH for the shell tool."""
    wrapper_dir = Path(tempfile.gettempdir()) / "skilldag-alfworld-bin"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / "skilldag"
    wrapper_body = (
        "#!/usr/bin/env bash\n"
        f'exec "{sys.executable}" -m skilldag "$@"\n'
    )
    if not wrapper_path.exists() or wrapper_path.read_text(encoding="utf-8") != wrapper_body:
        wrapper_path.write_text(wrapper_body, encoding="utf-8")
        wrapper_path.chmod(0o755)
    return wrapper_dir


def run_shell_command(
    command: str,
    *,
    graph_path: Path,
    skills_dir: Path,
    task_id: str,
) -> tuple[str, int]:
    """Execute one shell command. Return (stdout, return_code).

    stdout/stderr is truncated to ``COMMAND_OUTPUT_MAX_CHARS`` so a noisy
    command can't eat the agent's context. Concurrent online graph edits
    rely on atomic ``tmp+rename`` in ``save_graph_data``; no cross-process
    locking, as experiment concurrency rarely overlaps on edits.
    """
    if not command.strip():
        return "[command error] empty command", 1
    if "\n" in command or "\r" in command:
        return "[command error] multi-line commands are not allowed", 1

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return f"[command error] failed to parse command: {exc}", 1
    if not tokens:
        return "[command error] empty command", 1

    # Ablation: frozen graph mode — block all propose/edit-edge mutations.
    if os.environ.get("SKILLDAG_FROZEN_MODE") == "1":
        try:
            idx_graph = tokens.index("graph")
            sub = tokens[idx_graph + 1] if idx_graph + 1 < len(tokens) else ""
        except ValueError:
            sub = ""
        if sub.startswith("propose-") or sub == "edit-edge":
            return (
                "[ablation:frozen-graph] graph mutation is disabled in this run. "
                "search / show / get-* are still available; choose one of those instead.",
                1,
            )

    wrapper_dir = _ensure_skilldag_wrapper()
    env = {
        **os.environ,
        "SKILLDAG_GRAPH_PATH": str(graph_path),
        "SKILLDAG_SKILLS_DIR": str(skills_dir),
        "SKILLDAG_TASK_ID": task_id,
        "PYTHONPATH": f"{SKILLDAG_SRC}:{SKILLDAG_HOME}:{os.environ.get('PYTHONPATH', '')}",
        "PATH": f"{wrapper_dir}:{os.environ.get('PATH', '')}",
    }

    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=str(SKILLDAG_HOME),
            env=env,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired:
        return f"[command error] timeout after {COMMAND_TIMEOUT_SECS}s", 124
    out = proc.stdout or ""
    if proc.stderr and proc.stderr.strip():
        out = out + ("\n[stderr] " if out else "[stderr] ") + proc.stderr.strip()
    if len(out) > COMMAND_OUTPUT_MAX_CHARS:
        out = out[:COMMAND_OUTPUT_MAX_CHARS] + f"\n... (truncated {len(out) - COMMAND_OUTPUT_MAX_CHARS} chars)"
    return out, proc.returncode


# --- LLM call --------------------------------------------------------------

@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def llm_call(
    messages: list[dict],
    *,
    model: str,
    api_base: str,
    api_key: str,
    timeout_sec: float = LLM_REQUEST_TIMEOUT_SECS,
) -> tuple[str, dict]:
    """litellm completion with anthropic ephemeral caching. Wrapped with retry
    Backoff
    schedule: 5s → 10s → 20s → 40s → 80s plus jitter, 5 tries total."""
    messages = add_anthropic_caching(messages, model)

    if is_openrouter_chat_model(model, api_base):
        payload = {
            "model": openrouter_payload_model(model),
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{api_base.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
                "http-referer": "https://github.com/skilldag",
                "x-title": "SkillDAG ALFWorld eval",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as r:
                response = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter chat completion failed: {e.code} {body}") from e

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = extract_chat_completion_content(message)
        usage = normalize_openrouter_usage(response.get("usage") or {})
        return content, usage

    # Route gpt-5.x-codex (and other reasoning-capable models) to /v1/responses.
    # Without this, openai/gpt-5.2-codex falls through to litellm chat/completions,
    # which silently drops `reasoning: {effort}` → mutation never triggers.
    _use_responses = (
        model.lower().startswith("openai-responses/")
        or (model.lower().startswith("openai/") and "codex" in model.lower())
    )
    if _use_responses:
        # OpenAI /v1/responses path. System message → top-level instructions.
        sys_msg = next((m for m in messages if m.get("role") == "system"), None)
        instructions = sys_msg["content"] if sys_msg else ""
        if isinstance(instructions, list):
            instructions = "".join(b.get("text", "") for b in instructions if b.get("type") == "text")
        non_sys = [m for m in messages if m.get("role") != "system"]
        # input items: keep {role, content} as plain strings (Responses API accepts this)
        input_items = []
        for m in non_sys:
            c = m.get("content", "")
            if isinstance(c, list):
                c = "".join(b.get("text", "") for b in c if b.get("type") == "text")
            input_items.append({"role": m["role"], "content": c})
        payload = {
            "model": model.split("/", maxsplit=1)[1],
            "instructions": instructions,
            "input": input_items,
            "max_output_tokens": 2048,
            "reasoning": {"effort": "high"},  # SkillDAG: enables multi-step reflection for propose/edit-edge
            "store": False,
        }
        # Reasoning models (gpt-5.x-codex etc.) reject `temperature`; only set it for non-reasoning chat models
        if "codex" not in payload["model"].lower() and not payload["model"].lower().startswith("gpt-5"):
            payload["temperature"] = 0
        # Strip trailing /v1 so we don't double-append (api_base may be
        # "https://api.openai.com" or "https://api.openai.com/v1").
        _base = api_base.rstrip('/')
        if _base.endswith('/v1'):
            _base = _base[:-3]
        req = urllib.request.Request(
            f"{_base}/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as r:
                response = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI responses request failed: {e.code} {body}") from e

        # parse output: response.output[*].content[*].text where type=output_text
        text_parts = []
        for item in response.get("output", []):
            for blk in item.get("content", []):
                if blk.get("type") == "output_text":
                    text_parts.append(blk.get("text", ""))
        content = "".join(text_parts)
        usage_obj = response.get("usage") or {}
        usage = {
            "prompt_tokens": usage_obj.get("input_tokens", 0) or 0,
            "completion_tokens": usage_obj.get("output_tokens", 0) or 0,
            "cache_read": (usage_obj.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0,
            "cache_create": 0,
        }
        return content, usage

    if model.lower().startswith("anthropic/minimax"):
        payload = {
            "model": model.split("/", maxsplit=1)[1],
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{api_base.rstrip('/')}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as r:
                response = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed: {e.code} {body}") from e

        text_blocks = [
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text" and block.get("text")
        ]
        content = "".join(text_blocks)
        usage_obj = response.get("usage") or {}
        usage = {
            "prompt_tokens": usage_obj.get("input_tokens", 0) or 0,
            "completion_tokens": usage_obj.get("output_tokens", 0) or 0,
            "cache_read": usage_obj.get("cache_read_input_tokens", 0) or 0,
            "cache_create": usage_obj.get("cache_creation_input_tokens", 0) or 0,
        }
        return content, usage

    response = litellm.completion(
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout_sec,
        temperature=0.0,
        drop_params=True,
    )
    content = response.choices[0].message.content or ""
    usage_obj = getattr(response, "usage", None)
    input_tokens = getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0
    completion_tokens = getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0
    cache_read_tokens = getattr(usage_obj, "cache_read_input_tokens", 0) or 0 if usage_obj else 0
    cache_write_tokens = getattr(usage_obj, "cache_creation_input_tokens", 0) or 0 if usage_obj else 0
    usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": completion_tokens,
        "cache_read": cache_read_tokens,
        "cache_create": cache_write_tokens,
    }
    return content, usage


# --- single task -----------------------------------------------------------

def run_task(
    env,
    task_ob: str,
    task_name: str,
    args,
    runtime: SkillDAGRuntime | None,
) -> dict:
    task_id = (task_name.split("/")[0] or task_name)[:60] if task_name else "alfworld_task"

    messages: list[dict] = [{"role": "system", "content": _alfworld_system_prompt()}]
    if runtime is not None:
        header = _skilldag_task_header(task_id, runtime)
        messages.append({"role": "user", "content": header})
    messages.append({"role": "user", "content": task_ob})

    task_done = False
    task_reward = 0
    env_steps = 0            # counted ONLY when an env action is executed
    turns = 0                # counted on env-action attempts (env-step + parse-fail); CLI calls excluded
    cli_calls = 0            # counted ONLY when a skilldag CLI command is dispatched
    total_usage = {
        "prompt": 0,
        "completion": 0,
        "cache_read": 0,
        "cache_create": 0,
    }
    agent_edits: list[dict] = []   # entries for mutation CLI invocations
    cli_invocations: list[dict] = []   # all CLI calls (read + mutate) for audit
    consulted_skills: set[str] = set()
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    max_env_steps = args.max_steps
    max_turns = args.max_turns if args.max_turns > 0 else max_env_steps * 2
    max_cli_calls = 100

    while (not task_done and env_steps < max_env_steps and turns < max_turns
           and cli_calls < max_cli_calls):
        try:
            response, usage = llm_call(
                messages,
                model=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
            )
            total_usage["prompt"] += usage["prompt_tokens"]
            total_usage["completion"] += usage["completion_tokens"]
            total_usage["cache_read"] += usage["cache_read"]
            total_usage["cache_create"] += usage["cache_create"]
        except Exception as exc:
            print(f"{Colors.RED}LLM error turn {turns}: {exc}{Colors.RESET}")
            break

        response_raw = response
        response = sanitize_response(response)
        if response != response_raw:
            print(f"{Colors.BLUE}[sanitize] trimmed {len(response_raw) - len(response)} chars"
                  f"{Colors.RESET}")
        print(f"{Colors.GREEN}Agent (turn {turns+1} / env_step {env_steps+1} / cli {cli_calls}):\n{response}{Colors.RESET}")
        messages.append({"role": "assistant", "content": response})

        turn_payload, parse_error = parse_turn_payload(response)
        if parse_error:
            turns += 1
            messages.append({"role": "user", "content": (
                "[runner] " + parse_error + " "
                "Re-send as one JSON object with `thought` and exactly one of `action` or `command`."
            )})
            continue

        command = turn_payload["command"]
        action = turn_payload["action"]

        if command:
            if runtime is None:
                turns += 1
                messages.append({"role": "user", "content": (
                    "[runner] Shell command mode is disabled because the "
                    "SkillDAG runtime failed to initialize for this task."
                )})
                continue
            stdout, rc = run_shell_command(
                command,
                graph_path=runtime.graph_path,
                skills_dir=runtime.skills_dir,
                task_id=task_id,
            )
            cli_calls += 1
            try:
                cli_tokens = shlex.split(command)
            except ValueError:
                cli_tokens = []
            is_mut = _is_mutation_command(cli_tokens)
            skill_id = _extract_skill_id(command)
            if skill_id:
                consulted_skills.add(skill_id)
            record = {
                "turn": turns, "cli_call": cli_calls,
                "command": command, "rc": rc,
                "is_mutation": is_mut,
                "stdout_excerpt": stdout[:500],
            }
            cli_invocations.append(record)
            if is_mut:
                agent_edits.append(record)
                color = Colors.BLUE if rc == 0 else Colors.RED
                print(f"{color}[graph-mutation cli={cli_calls} rc={rc}] {command}{Colors.RESET}")
            else:
                print(f"{Colors.BLUE}[command cli={cli_calls} rc={rc}] {command[:120]}{Colors.RESET}")
            messages.append({"role": "user", "content": f"[command rc={rc}]\n{stdout}"})
            continue

        # Environment action
        turns += 1
        env_steps += 1
        obs_arr, _, done_arr, info = env.step([action])
        ob = process_ob(obs_arr[0])
        task_reward = info["won"][0]
        task_done = done_arr[0]
        print(f"{Colors.YELLOW}Obs: {ob}{Colors.RESET}")
        messages.append({"role": "user", "content": f"Observation: {ob}"})

        if task_done:
            break

    cli_budget_exhausted = cli_calls >= max_cli_calls and not task_done
    if cli_budget_exhausted:
        print(f"{Colors.RED}[runner] cli budget exhausted at {cli_calls}/{max_cli_calls}; "
              f"task aborted (env_steps={env_steps}, turns={turns}){Colors.RESET}")

    runtime_sec = round(time.perf_counter() - t0, 3)
    loaded_skills = sorted(consulted_skills)

    return {
        "query": task_ob.split("Your task is to: ")[-1].split("\n")[0].strip(),
        "name": task_name,
        "task_done": task_done,
        "reward": task_reward,
        "steps": env_steps,
        "turns": turns,
        "cli_calls": cli_calls,
        "cli_budget_exhausted": cli_budget_exhausted,
        "messages": messages,
        "loaded_skills": loaded_skills,
        "cli_invocations": cli_invocations,
        "agent_edits": agent_edits,
        "token_usage": total_usage,
        "started_at": started_at,
        "agent_runtime_seconds": runtime_sec,
    }


# --- per-process eval entry -------------------------------------------------

def eval_single_game(game_idx: int, args, config: dict, split: str, output_path: str):
    env = None
    try:
        env = get_environment(config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=1)

        # Seek by reset(); ALFWorld TWEnv rotates through gamefiles round-robin.
        for _ in range(game_idx + 1):
            obs_list, info = env.reset()

        game_name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        runtime = SkillDAGRuntime(
            graph_path=args.graph_path,
            skills_dir=args.skills_dir,
            task_id=game_name,
        )

        ob_str = "\n".join(obs_list[0].split("\n\n")[1:])
        result = run_task(env, ob_str, game_name, args, runtime)
        attach_alfworld_skill_use_metric(result, alfworld_data=Path(args.alfworld_data))

        save_file = f"{output_path}/idx_{game_idx}.json"
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        return result
    except Exception as exc:
        print(f"{Colors.RED}Error game {game_idx}: {exc}{Colors.RESET}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


# --- bootstrap helper -------------------------------------------------------

def ensure_graph(graph_path: Path, skills_dir: Path, force: bool = False) -> None:
    """Ensure skillgraph.json exists at graph_path; bootstrap if missing."""
    if graph_path.exists() and not force:
        return
    print(f"Bootstrapping graph {graph_path} from {skills_dir}")
    env = os.environ.copy()
    env["SKILLDAG_BOOTSTRAP_ENABLED"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "skilldag",
            "initialize-graph",
            "--graph-path",
            str(graph_path),
            "--skills-dir",
            str(skills_dir),
            *(["--force"] if force else []),
        ],
        check=True,
        env=env,
        cwd=str(SKILLDAG_HOME),
    )


# --- entry point from alfworld_run.py --mode skilldag ---------------------

def _resolve_args_from_upstream(upstream_args) -> SimpleNamespace:
    """Map the upstream alfworld_run.py argparse Namespace to the fields the
    SkillDAG runner expects. Upstream-defined skilldag flags (prefixed with
    ``skilldag_``) take precedence over shared flags where appropriate.
    """
    # Prefer explicit skilldag_model; fall back to upstream --model only if the
    # user didn't leave it at its upstream default of 'gpt-4o'.
    model = getattr(upstream_args, "skilldag_model", None)
    if not model:
        model = upstream_args.model if upstream_args.model != 'gpt-4o' else DEFAULT_MODEL
    api_base = getattr(upstream_args, "skilldag_api_base", None) or DEFAULT_API_BASE
    if is_openrouter_chat_model(model, api_base):
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("API_KEY")
            or ""
        )
    else:
        api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("API_KEY")
            or os.environ.get('OPENROUTER_API_KEY')
            or ''
        )
    max_turns = getattr(upstream_args, "skilldag_max_turns", 0) or 0

    return SimpleNamespace(
        # LLM
        model=model,
        api_base=api_base,
        api_key=api_key,
        # Run
        config=DEFAULT_BASE_CONFIG,
        alfworld_data=os.environ.get("ALFWORLD_DATA", DEFAULT_ALFWORLD_DATA),
        split=upstream_args.split,
        max_workers=upstream_args.max_workers,
        max_steps=upstream_args.max_steps,
        max_turns=max_turns,
        max_games=upstream_args.max_games,
        task_indices=getattr(upstream_args, "task_indices", None),
        exp_name=upstream_args.exp_name or "skilldag",
        # SkillDAG
        graph_path=getattr(upstream_args, "skilldag_graph", None) or DEFAULT_GRAPH_PATH,
        skills_dir=upstream_args.skills_dir or DEFAULT_SKILLS_DIR,
        auto_bootstrap=getattr(upstream_args, "skilldag_auto_bootstrap", False),
        force_bootstrap=getattr(upstream_args, "skilldag_force_bootstrap", False),
    )


def run_skilldag(upstream_args) -> None:
    """Dispatched from alfworld_run.py when ``--mode skilldag``."""
    args = _resolve_args_from_upstream(upstream_args)

    if not args.api_key:
        print(
            f"{Colors.RED}WARN: no api_key set "
            f"(OPENAI_API_KEY / OPENROUTER_API_KEY / API_KEY / OPENAI_API_KEY env){Colors.RESET}"
        )

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    alf_data = Path(args.alfworld_data).resolve()
    if not alf_data.exists():
        raise FileNotFoundError(f"ALFWORLD_DATA dir not found: {alf_data}")

    config["dataset"]["data_path"] = str(alf_data / "json_2.1.1" / "train")
    config["dataset"]["eval_id_data_path"] = str(alf_data / "json_2.1.1" / "valid_seen")
    config["dataset"]["eval_ood_data_path"] = str(alf_data / "json_2.1.1" / "valid_unseen")
    config["logic"]["domain"] = str(alf_data / "logic" / "alfred.pddl")
    config["logic"]["grammar"] = str(alf_data / "logic" / "alfred.twl2")
    # mrcnn: only used by ThorEnv; safe to set even if file name is different
    config.setdefault("mask_rcnn", {})
    # Fix the rl.training nesting required by alfworld init_env
    config.setdefault("rl", {}).setdefault("training", {})["max_nb_steps_per_episode"] = args.max_steps

    if args.split == "train":
        split = "train"
    elif args.split == "dev":
        split = "eval_in_distribution"
    else:
        split = "eval_out_of_distribution"
    output_path = SKILLDAG_HOME / "results" / "alfworld" / args.exp_name
    output_path.mkdir(parents=True, exist_ok=True)

    if args.auto_bootstrap or not Path(args.graph_path).exists():
        ensure_graph(Path(args.graph_path), Path(args.skills_dir), force=args.force_bootstrap)

    # Count total games in the chosen split
    temp_env = get_environment(config["env"]["type"])(config, train_eval=split)
    temp_env = temp_env.init_env(batch_size=1)
    num_games = len(temp_env.gamefiles)
    try:
        temp_env.close()
    except Exception:
        pass
    del temp_env

    # Resume: skip idxs already in output_path
    existing: set[int] = set()
    for f in output_path.glob("idx_*.json"):
        try:
            existing.add(int(f.stem.split("_")[1]))
        except ValueError:
            continue

    # Indices selection — task_indices wins (matches alfworld_run.py:446-449 semantics).
    # task_indices may be a CSV string (from --task_indices) or already a list.
    if args.task_indices:
        if isinstance(args.task_indices, str):
            requested = [int(x.strip()) for x in args.task_indices.split(",") if x.strip()]
        else:
            requested = list(args.task_indices)
        indices = [i for i in requested if 0 <= i < num_games]
        if len(indices) != len(requested):
            dropped = [i for i in requested if not (0 <= i < num_games)]
            print(
                f"{Colors.YELLOW}WARN: dropped {len(dropped)} task_indices outside "
                f"[0,{num_games}): {dropped[:10]}{'...' if len(dropped)>10 else ''}{Colors.RESET}"
            )
    else:
        indices = list(range(num_games))
    if args.max_games is not None:
        indices = indices[: args.max_games]
    tasks_to_run = [i for i in indices if i not in existing]

    print(
        f"\n=== ALFWorld SkillDAG run ===\n"
        f"  split            : {split}\n"
        f"  total games      : {num_games}\n"
        f"  to_run           : {len(tasks_to_run)}\n"
        f"  max_workers      : {args.max_workers}\n"
        f"  max_steps (env)  : {args.max_steps}\n"
        f"  max_turns (env-attempt)  : {args.max_turns} (0 ⇒ 2× max_steps)\n"
        f"  model            : {args.model}\n"
        f"  graph_path       : {args.graph_path}\n"
        f"  skills_dir       : {args.skills_dir}\n"
    )

    if not tasks_to_run:
        print("Nothing to run.")
        metrics_payload = write_alfworld_skill_use_metrics(
            [output_path],
            output=output_path / "skill_use_metrics.json",
            alfworld_data=alf_data,
        )
        print("\n=== ALFWorld SkillDAG skill-use correctness ===")
        print_skill_use_summary(metrics_payload)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_idx = {
            ex.submit(eval_single_game, idx, args, config, split, str(output_path)): idx
            for idx in tasks_to_run
        }
        total_reward = 0.0
        done_count = 0
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                res = fut.result()
                if res is not None:
                    done_count += 1
                    total_reward += float(res.get("reward") or 0)
                    avg = total_reward / done_count if done_count else 0
                    print(
                        f"[{done_count}/{len(tasks_to_run)}] idx={idx} "
                        f"reward={res.get('reward')} steps={res.get('steps')} "
                        f"avg_R={avg:.3f}"
                    )
            except Exception as exc:
                print(f"game {idx} error: {exc}")

    metrics_payload = write_alfworld_skill_use_metrics(
        [output_path],
        output=output_path / "skill_use_metrics.json",
        alfworld_data=alf_data,
    )
    print("\n=== ALFWorld SkillDAG skill-use correctness ===")
    print_skill_use_summary(metrics_payload)
