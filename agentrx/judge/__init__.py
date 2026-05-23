from typing import Any, Dict, List, Optional, Tuple
import json, os, re, time

Event = Dict[str, Any]
TRAJECTORY_IR_SCHEMA: Event = {
    "trajectory_id": "str",
    "instruction": "str",
    "steps": [
        {
            "index": "int",
            "substeps?": [
                {"sub_index": "int", "role": "str", "content": "str"}
            ],
        }
    ],
}

# TODO -- add ground truth, system prompt parameter?
PREFERRED_KEYS = ("traj", "events", "messages", "trajectory", "spans")
ID_KEYS = ("trajectory_id", "task_id", "traceId")

Trajectory = List[Dict[str, Any]]  # {"trajectory_id": str, "events": List[Event]}

def find_first_key(obj: Any, key: str, max_depth: int = 6) -> Optional[Any]:
    """
    Return the first value found for an exact dict key match, searching nested dict/list.
    (No normalization, no heuristics. Exact key only.)
    """
    if max_depth < 0:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return obj.get(key)
        for v in obj.values():
            hit = find_first_key(v, key, max_depth=max_depth - 1)
            if hit is not None:
                return hit
        return None
    if isinstance(obj, list):
        for it in obj:
            hit = find_first_key(it, key, max_depth=max_depth - 1)
            if hit is not None:
                return hit
        return None
    return None

def extract_instruction(container: Any, events: List[Event]) -> str:
    v = find_first_key(container, "instruction") or find_first_key(container, "task")
    return str(v).strip() if v is not None else ""

def _parse_markdown_trajectory(raw: str, default_tid: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a markdown conversation trajectory into the standard events format.

    Expected format:
      # User Properties
      - **key**: value
      ...

      # Conversation
      ## ***user*** #0
      <content>
      <hr style="border:5px solid">
      ## ***assistant*** #1
      <content>
      ...

    Returns None if the content doesn't look like a markdown conversation.
    Returns data with a special ``_markdown_ir`` flag so that ``run_ir`` can
    skip the domain converter and use the pre-built IR directly.
    """
    # Quick check: must contain the markdown turn pattern
    TURN_RE = re.compile(
        r"^##\s+\*{3}(\w+)\*{3}\s+#(\d+)\s*$", re.MULTILINE
    )
    turns = list(TURN_RE.finditer(raw))
    if not turns:
        return None

    # Extract metadata from "# User Properties" section
    metadata: Dict[str, str] = {}
    prop_re = re.compile(r"^-\s+\*\*(\w[\w_]*)\*\*:\s*(.+)$", re.MULTILINE)
    for m in prop_re.finditer(raw):
        metadata[m.group(1)] = m.group(2).strip()

    tid = metadata.get("scenario_name", default_tid)
    instruction = metadata.get("first_turn_prompt", "")

    # Parse conversation turns
    events: List[Dict[str, Any]] = []
    for idx, match in enumerate(turns):
        role = match.group(1).lower()  # "user" or "assistant"
        # Content runs from end of this header to start of next turn (or end of file)
        start = match.end()
        end = turns[idx + 1].start() if idx + 1 < len(turns) else len(raw)
        content = raw[start:end].strip()
        # Strip trailing <hr> separators
        content = re.sub(r'<hr\s+style="[^"]*">\s*$', "", content).strip()
        # Skip empty sentinel turns like "conversation_end_marker"
        if content == "conversation_end_marker":
            continue
        events.append({"role": role, "content": content})

    return [{"trajectory_id": tid, "instruction": instruction, "events": events,
             "metadata": metadata, "_markdown_ir": True}]


def markdown_ir(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert markdown-parsed trajectories (role/content events) into canonical IR.
    Each conversation turn becomes its own step (1-based index).
    """
    out: List[Dict[str, Any]] = []
    for t in trajectories:
        trajectory_id = str(t.get("trajectory_id") or "unknown")
        instruction = str(t.get("instruction") or "")
        messages: List[Dict[str, Any]] = t.get("events", []) or []

        steps: List[Dict[str, Any]] = []
        for i, m in enumerate(messages):
            role = str(m.get("role") or "unknown")
            content = str(m.get("content") or "")
            steps.append({
                "index": i + 1,
                "substeps": [{"sub_index": 1, "role": role, "content": content}],
            })

        ir = {"trajectory_id": trajectory_id, "instruction": instruction, "steps": steps}
        validate_ir(ir)
        out.append(ir)
    return out


def load_trajectories(path: str) -> List[Trajectory]:
    """
    Read a file and normalize it into a list of:
      [{"trajectory_id": str, "events": list[dict]}, ...]
    
    Supports:
    A) Whole-file JSON:
         1) dict wrapper: {"events":[...]} / {"traj":[...]} / {"messages":[...]} / {"trajectory":[...]}
         2) list of events -> Single Trajectory. List[Dict[str, Any]] where each element is an event dict and none of them is a wrapper dict (i.e., none contains "events"|"traj"|...).
              [ {...}, {...} ], the trajectory_id is generally the filename
         3) list of wrappers: List[Dict[str, Any]] where each element is a wrapper dict containing one of the wrapper keys mapping to List[Dict]
[
  {
    "trajectory_id": "T1",
    "events": [
      {"source": "user", "message": "hi"}
    ]
  },
  {
    "task_id": "T2",
    "messages": [
      {"source": "user", "message": "yo"}
    ]
  }
]

    B) Streamed objects / JSONL 
        1) Event stream -> Single trajectory. a sequence of Event dicts: {...}\n{...}\n{...} Example:
            {"source":"user","message":"hi"}\n{"source":"assistant","message":"yo"}
        2) Wrapper stream -> Many trajectories. a sequence of Wrapper dicts: {...}\n{...}\n{...}
            Example: {"trajectory_id":"A", "events":[...]}\n{"trajectory_id":"B", "events":[...]}
    """

    def filename_stem() -> str:
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        return stem or "trajectory"

    # extract an ID from a dict using known keys.
    def extract_id(d: Dict[str, Any]) -> Optional[str]:
        for k in ID_KEYS:
            v = d.get(k)
            if v is not None:
                s = str(v).strip()
                if s:
                    return s
        return None

    # Sometimes the wrapper doesn't have the ID, but some events do. So this scans through the event dicts and tries to find trajectory_id/task_id.
    def extract_id_from_events(events: List[Event]) -> Optional[str]:
        for e in events:
            if isinstance(e, dict):
                tid = extract_id(e)
                if tid:
                    return tid
        return None

    def extract_events(obj: Dict[str, Any]) -> List[Event]:
    # -------------------------------------------------------------------------
    # Helper: given a dict "obj", extract the *events list* out of it.
    #
    # PREFERRED_KEYS is defined outside:
    #   PREFERRED_KEYS = ("traj", "events", "messages", "trajectory")
    #
    # Logic:
    #   - If obj has one of those keys mapping to a list, interpret that list as events.
    #   - Validate that every element in that list is a dict (event dict).
    #   - If no preferred key exists, treat obj itself as a single "event".
    #
    # Why the fallback [obj]?
    #   Because sometimes a JSON file might contain a *single event dict*,
    #   and we still want to normalize it into events=[that_one_event].
    # -------------------------------------------------------------------------
        for k in PREFERRED_KEYS:
            v = obj.get(k)
            if isinstance(v, list):
                if not all(isinstance(x, dict) for x in v):
                    raise ValueError(f'"{k}" must be list[dict].')
                return v  # type: ignore[return-value]
        return [obj]


    raw = open(path, "r", encoding="utf-8-sig").read().strip()
    if not raw:
        return [{"trajectory_id": filename_stem(), "instruction": "", "events": []}]

    default_tid = filename_stem()

    try:
        obj = json.loads(raw)

        # Case A1: whole file is a dict
        if isinstance(obj, dict):
            events = extract_events(obj)
            tid = extract_id(obj) or extract_id_from_events(events) or default_tid
            instr = extract_instruction(obj, events)
            return [{"trajectory_id": tid, "instruction": instr, "events": events}]

        # Case A2 / A3: whole file is a list
        if isinstance(obj, list):
            if not obj:
                return [{"trajectory_id": tid, "instruction": instr, "events": events}]

            # list[dict] events => single trajectory
            if all(isinstance(x, dict) and not any(k in x for k in PREFERRED_KEYS) for x in obj):
                events = obj  # type: ignore[assignment]
                tid = extract_id_from_events(events) or default_tid
                instr = extract_instruction(obj, events)
                return [{"trajectory_id": tid, "instruction": instr, "events": events}]

            # list[dict] wrappers => many trajectories
            if all(isinstance(x, dict) and any(k in x for k in PREFERRED_KEYS) for x in obj):
                out: List[Trajectory] = []
                for idx, w in enumerate(obj):
                    events = extract_events(w)
                    tid = extract_id(w) or extract_id_from_events(events) or f"{default_tid}__{idx+1}"
                    instr = extract_instruction(w, events)
                    out.append({"trajectory_id": tid, "instruction": instr, "events": events})
                return out

            raise ValueError("Unrecognized JSON list shape (mixed wrappers/events).")

        raise ValueError("JSON must be a dict or a list.")

    except json.JSONDecodeError:
        pass

    # --- Markdown conversation fallback ---
    md_result = _parse_markdown_trajectory(raw, default_tid)
    if md_result is not None:
        return md_result

    # --- Streamed / JSONL / glued-object fallback ---
    dec = json.JSONDecoder()
    i, n = 0, len(raw)
    objs: List[Dict[str, Any]] = []
    while True:
        while i < n and raw[i].isspace():
            i += 1
        if i >= n:
            break

        # Parse one JSON value starting at position i.
        o, i = dec.raw_decode(raw, i)
        if not isinstance(o, dict):
            raise ValueError("Stream must contain JSON objects (dict).")
        objs.append(o)

    if not objs:
        return [{"trajectory_id": default_tid, "instruction": "", "events": []}]

    # Decide whether each streamed dict "looks like a wrapper":
    # wrapper means: it has a preferred key AND that key's value is a list.
    are_wrappers = [any(k in o and isinstance(o.get(k), list) for k in PREFERRED_KEYS) for o in objs]

    # B2) If all objects are wrappers => many trajectories.
    if all(are_wrappers):
        out: List[Trajectory] = []
        for idx, w in enumerate(objs):
            events = extract_events(w)
            tid = extract_id(w) or extract_id_from_events(events) or f"{default_tid}__{idx+1}"
            instr = extract_instruction(w, events)
            out.append({"trajectory_id": tid, "instruction": instr, "events": events})
        return out  # many trajectories

    # B1) If none are wrappers => it's an event stream => single trajectory.
    if not any(are_wrappers):
        events = objs
        tid = extract_id_from_events(events) or default_tid
        instr = extract_instruction(objs, events)
        return [{"trajectory_id": tid, "instruction": instr, "events": events}] # single trajectory

    raise ValueError("Mixed wrapper-objects and event-objects in the same stream.")

def validate_ir(ir: Event) -> None:
    """
    Minimal validation for the IR shape described above.

    Convention in this implementation:
      - step.index is 1-based (must be >= 1)
      - substep.sub_index is 1-based (must be >= 1)
    """

    if not isinstance(ir, dict):
        raise ValueError("IR must be a dict")
    if "trajectory_id" not in ir or "steps" not in ir:
        raise ValueError("IR must have trajectory_id and steps")
    if not isinstance(ir["steps"], list):
        raise ValueError("IR.steps must be a list")
    if "trajectory_id" not in ir or "instruction" not in ir or "steps" not in ir:
        raise ValueError("IR must have trajectory_id, instruction, and steps")
    if not isinstance(ir["instruction"], str):
        raise ValueError("IR.instruction must be str")
    for s in ir["steps"]:
        if not isinstance(s, dict):
            raise ValueError("Each step must be a dict")
        if "index" not in s:
            raise ValueError("Step missing key: index")
        if not isinstance(s["index"], int):
            raise ValueError("Step.index must be int")
        if s["index"] < 0:
            raise ValueError("Step.index must be >= 0 (domain may be 0-based or 1-based)")

        subs = s.get("substeps", [])
        if not isinstance(subs, list):
            raise ValueError("Step.substeps must be list")

        for sub in subs:
            if not isinstance(sub, dict):
                raise ValueError("Each substep must be a dict")
            if "substeps" in sub:
                raise ValueError("Substeps cannot contain substeps (no recursion)")
            for k in ("sub_index", "role", "content"):
                if k not in sub:
                    raise ValueError(f"Substep missing key: {k}")
            if not isinstance(sub["sub_index"], int):
                raise ValueError("Substep.sub_index must be int")
            if sub["sub_index"] < 1:
                raise ValueError("Substep.sub_index must start at 1")
            if not isinstance(sub["role"], str) or not isinstance(sub["content"], str):
                raise ValueError("Substep.role/content must be str")


def is_ir(data: list) -> bool:
    """Return True if *data* already looks like a list of valid IR dicts."""
    if not data:
        return False
    try:
        validate_ir(data[0])
        return True
    except (ValueError, TypeError, KeyError):
        return False


def ensure_ir(path: str, domain: str) -> list:
    """Load *path* and return IR.  Converts on the fly if the file is raw."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw_json = json.load(f)
    data = raw_json if isinstance(raw_json, list) else [raw_json]

    if is_ir(data):
        return data

    # Not IR yet – run the domain converter
    from agentrx.invariants.domain_registry import get_domain_config
    raw = load_trajectories(path)
    converted = get_domain_config(domain).ir_converter(raw)
    if not isinstance(converted, list):
        converted = [converted]
    return converted


def tau_bench_ir(trajectories: List[Trajectory]) -> List[Event]:
    """
    tau-bench style: events are typically chat messages like:
      {"role": "user"|"assistant"|"system", "content": "..."}   (or "message")

    Changes vs prior version:
      - If role == "tool": include ONLY the tool payload content (no tool_call_id, no name, no wrappers).
      - If role == "assistant" and it has "tool_calls": dump the ENTIRE tool_calls field as a JSON string
        into content (no custom pretty formatting blocks).
    """
    out: List[Event] = []

    def _pretty_json_if_possible(s: str) -> str:
        s2 = (s or "").strip()
        if not s2:
            return ""
        if (s2.startswith("{") and s2.endswith("}")) or (s2.startswith("[") and s2.endswith("]")):
            try:
                return json.dumps(json.loads(s2), ensure_ascii=False, indent=2)
            except Exception:
                return s
        return s

    for t in trajectories:
        trajectory_id = str(t.get("trajectory_id", "unknown"))
        instruction = str(t.get("instruction") or "")
        messages: List[Dict[str, Any]] = t.get("events", []) or []

        steps: List[Dict[str, Any]] = []
        for i, m in enumerate(messages):
            role = (m.get("role") or "").strip() or "unknown"

            raw_content = m.get("content")
            if raw_content is None:
                raw_content = m.get("message", "")

            parts: List[str] = []

            # 1) Base content
            if role == "tool":
                payload = "" if raw_content is None else str(raw_content)
                pretty_payload = _pretty_json_if_possible(payload).strip()
                if pretty_payload:
                    parts.append(pretty_payload)
            else:
                s = str(raw_content).strip()
                if s:
                    parts.append(s)

            # 2) Tool calls (assistant: dump whole field as JSON string)
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                if role == "assistant":
                    try:
                        parts.append(json.dumps(tool_calls, ensure_ascii=False, indent=2, sort_keys=True))
                    except Exception:
                        parts.append(str(tool_calls))
                else:
                    # Non-assistant tool_calls are rare; keep a simple readable fallback.
                    try:
                        parts.append(json.dumps(tool_calls, ensure_ascii=False, indent=2, sort_keys=True))
                    except Exception:
                        parts.append(str(tool_calls))

            # 3) Optional extra fields (kept as-is)
            if "function" in m:
                fs = str(m.get("function") or "").strip()
                if fs:
                    parts.append(f"[function]\n{fs}")
            if "response" in m:
                rs = str(m.get("response") or "").strip()
                if rs:
                    parts.append(f"[response]\n{rs}")

            content = "\n\n".join(parts).strip()
            steps.append(
                {
                    "index": i+1,
                    "substeps": [{"sub_index": 1, "role": role, "content": str(content)}],
                }
            )

        ir = {"trajectory_id": trajectory_id, "instruction": instruction, "steps": steps}
        validate_ir(ir)
        out.append(ir)

    return out

def flash_ir(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flash/orchestrator style:
      - Each trajectory has "events": list[event_dict]
      - We ignore events with type == "LLMCallEvent"
      - We create "steps" as containers and each event becomes a substep inside the current step
      - We only switch current step when we see an explicit step marker

    Step marker detection:
      - role/source contains something like "(Step-3)" or "Step 3"
      - OR content begins with something like "Step-3 ..."
    """
    STEP_RE = re.compile(r"\bStep[-\s]*(\d+)\b", re.IGNORECASE)

    def step_num_from(role: str, content: str):
        m = STEP_RE.search(role or "")
        if m:
            return int(m.group(1))
        m = STEP_RE.search((content or "").lstrip())
        return int(m.group(1)) if m else None

    out: List[Dict[str, Any]] = []

    for t in trajectories:
        trajectory_id = str(t.get("trajectory_id") or "unknown")
        events = t["events"]
        instruction = next((str((e.get("message") if e.get("message") is not None else e.get("content","")) or "") for e in events if e.get("type") != "LLMCallEvent" and str(e.get("source") or "").strip() == "Orchestrator (thought)" and str((e.get("message") if e.get("message") is not None else e.get("content","")) or "").lstrip().startswith("Initial plan:")), "")

        steps: List[Dict[str, Any]] = []
        by_index: Dict[int, Dict[str, Any]] = {}

        def ensure_step(i: int) -> Dict[str, Any]:
            s = by_index.get(i)
            if s is None:
                s = {"index": i, "substeps": []}
                by_index[i] = s
                steps.append(s)
            return s

        current = ensure_step(1)

        for e in events:
            if e.get("type") == "LLMCallEvent":
                continue

            role = str(e.get("source") or "unknown")
            content = e.get("message")
            if content is None:
                content = e.get("content", "")
            content = str(content)

            # Only switch steps on actual step markers
            if ("(Step-" in role) or content.lstrip().startswith("Step-"):
                n = step_num_from(role, content)
                if n is not None:
                    current = ensure_step(n)

            current["substeps"].append({
                "sub_index": len(current["substeps"]) + 1,
                "role": role,
                "content": content,
            })

        steps.sort(key=lambda s: s["index"])
        ir = {"trajectory_id": trajectory_id, "steps": steps, "instruction": instruction}
        validate_ir(ir)
        out.append(ir)

    return out

def magentic_ir(trajectories: List[Trajectory]) -> List[Event]:
    """
    Magentic-one JSON trace style:
      events are list[{"role": str, "content": str}, ...]

    We set instruction = first "human" message content (if any),
    and each event becomes one step with a single substep.
    """
    out: List[Event] = []

    for t in trajectories:
        trajectory_id = str(t.get("trajectory_id") or "unknown")
        events: List[Dict[str, Any]] = t.get("events", []) or []

        # instruction = first human/user message (best-effort)
        instruction = ""
        for e in events:
            role = str(e.get("role") or "").strip().lower()
            if role in ("human", "user"):
                instruction = str(e.get("content") or "").strip()
                break

        steps: List[Dict[str, Any]] = []
        for i, e in enumerate(events):
            role = str(e.get("role") or "unknown")
            content = "" if e.get("content") is None else str(e.get("content"))

            steps.append({
                "index": i + 1,
                "substeps": [{"sub_index": 1, "role": role, "content": content}],
            })

        ir = {"trajectory_id": trajectory_id, "instruction": instruction, "steps": steps}
        validate_ir(ir)
        out.append(ir)

    return out


# ---------------------------------------------------------------------------
# LLM-BASED IR CONVERTER  (for new / unknown domains)
# ---------------------------------------------------------------------------

_LLM_IR_SYSTEM_PROMPT = """\
You are an expert at converting raw agent trajectory logs into a standard
Intermediate Representation (IR) JSON format.

TARGET IR SCHEMA
----------------
{
  "trajectory_id": "<string>",
  "instruction": "<string — the user's original task / request>",
  "steps": [
    {
      "index": <int, 1-based>,
      "substeps": [
        {
          "sub_index": <int, 1-based within the step>,
          "role": "<string — who spoke, e.g. user / assistant / tool / Orchestrator>",
          "content": "<string — the raw text of that turn>"
        }
      ]
    }
  ]
}

RULES
-----
1. Every trajectory MUST produce exactly ONE JSON object matching the schema.
2. trajectory_id: use whatever ID is present in the raw data; if none, use the
   index "trajectory_<N>".
3. instruction: the user's original request / task description. If not obvious,
   use the first user/human message content.
4. steps: break the raw events into logical steps. Each step groups one or more
   substeps. If the raw log is a flat list of messages, each message becomes its
   own step with a single substep.
5. step.index starts at 1 and increments.
6. substep.sub_index starts at 1 within each step.
7. role: preserve the original speaker/agent/tool name. Normalise only
   whitespace.
8. content: keep the full original text. If the raw event is structured JSON
   (tool call args, tool results, etc.), serialise it as a readable JSON string.
9. Output ONLY valid JSON — no markdown fences, no commentary.
10. If the input contains multiple trajectories, return a JSON ARRAY of IR
    objects.

EXAMPLE (single flat-message trajectory)
-----------------------------------------
Raw input:
[
  {"role": "user", "content": "Cancel my order #W123"},
  {"role": "assistant", "content": "Sure, let me look that up."},
  {"role": "tool", "content": "{\\"order_id\\": \\"#W123\\", \\"status\\": \\"pending\\"}"}
]

Expected output:
{
  "trajectory_id": "trajectory_1",
  "instruction": "Cancel my order #W123",
  "steps": [
    {"index": 1, "substeps": [{"sub_index": 1, "role": "user", "content": "Cancel my order #W123"}]},
    {"index": 2, "substeps": [{"sub_index": 1, "role": "assistant", "content": "Sure, let me look that up."}]},
    {"index": 3, "substeps": [{"sub_index": 1, "role": "tool", "content": "{\\"order_id\\": \\"#W123\\", \\"status\\": \\"pending\\"}"}]}
  ]
}
"""


def llm_ir(
    trajectories: List[Dict[str, Any]],
    *,
    endpoint: str = "azure",
    max_retries: int = 5,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    LLM-based IR converter for domains that lack a hand-written converter.

    Sends each raw trajectory to an LLM with the IR schema + rules, parses
    the response, validates with validate_ir(), and retries (feeding the
    validation error back) until success or max_retries is exhausted.
    """
    import agentrx.pipeline.globals as g

    if endpoint == "copilot":
        from agentrx.llm_clients.copilot_cli import copilot_mk_client
        client = copilot_mk_client()
        model = "copilot-cli"
    elif endpoint == "azure":
        from agentrx.llm_clients.azure import LLMAgent as LLMAgentAzure
        client = LLMAgentAzure.azure_mk_client()
        model = g.DEPLOYMENT
    elif endpoint == "trapi":
        from agentrx.llm_clients.trapi import LLMAgent as LLMAgentTrapi
        client = LLMAgentTrapi.trapi_mk_client()
        model = g.TRAPI_DEPLOYMENT_NAME
    elif endpoint == "openrouter":
        from agentrx.llm_clients.openrouter import LLMAgent as LLMAgentOpenRouter
        client = LLMAgentOpenRouter.openrouter_mk_client()
        model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    else:
        raise ValueError(f"Unknown endpoint: {endpoint!r}. Expected 'copilot', 'azure', 'trapi', or 'openrouter'.")

    # Max chars for the raw trajectory JSON sent to the LLM.
    # ~200K tokens ≈ 800K chars, leaving room for system prompt + response.
    MAX_TRAJ_CHARS = 800_000

    results: List[Dict[str, Any]] = []

    for t_idx, traj in enumerate(trajectories):
        # Build the user message with the raw trajectory
        raw_json = json.dumps(traj, indent=2, ensure_ascii=False)

        # Truncate oversized trajectories to fit within the LLM context window
        if len(raw_json) > MAX_TRAJ_CHARS:
            if verbose:
                print(
                    f"[LLM-IR]   trajectory {t_idx} too large "
                    f"({len(raw_json)} chars) — truncating to {MAX_TRAJ_CHARS} chars",
                    flush=True,
                )
            # Try to truncate at the event/list-item level for cleaner output
            if isinstance(traj, list):
                # Binary search for the largest prefix that fits
                lo, hi = 1, len(traj)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if len(json.dumps(traj[:mid], indent=2, ensure_ascii=False)) <= MAX_TRAJ_CHARS:
                        lo = mid
                    else:
                        hi = mid - 1
                truncated = traj[:lo]
                raw_json = json.dumps(truncated, indent=2, ensure_ascii=False)
                if verbose:
                    print(
                        f"[LLM-IR]   kept {lo}/{len(traj)} events ({len(raw_json)} chars)",
                        flush=True,
                    )
            elif isinstance(traj, dict) and any(
                isinstance(traj.get(k), list) for k in PREFERRED_KEYS
            ):
                # Truncate the main event list inside the dict
                for k in PREFERRED_KEYS:
                    if isinstance(traj.get(k), list) and len(traj[k]) > 1:
                        events = traj[k]
                        lo, hi = 1, len(events)
                        while lo < hi:
                            mid = (lo + hi + 1) // 2
                            trial = {**traj, k: events[:mid]}
                            if len(json.dumps(trial, indent=2, ensure_ascii=False)) <= MAX_TRAJ_CHARS:
                                lo = mid
                            else:
                                hi = mid - 1
                        traj = {**traj, k: events[:lo]}
                        raw_json = json.dumps(traj, indent=2, ensure_ascii=False)
                        if verbose:
                            print(
                                f"[LLM-IR]   kept {lo}/{len(events)} items in '{k}' ({len(raw_json)} chars)",
                                flush=True,
                            )
                        break
            else:
                # Last resort: hard character truncation
                raw_json = raw_json[:MAX_TRAJ_CHARS]

        user_msg = (
            f"Convert the following raw trajectory (index {t_idx}) into the "
            f"standard IR format described in the system prompt.\n\n"
            f"RAW TRAJECTORY:\n{raw_json}"
        )

        messages = [
            {"role": "system", "content": _LLM_IR_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        ir_obj = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            if verbose:
                print(
                    f"[LLM-IR] trajectory {t_idx} attempt {attempt}/{max_retries} …",
                    flush=True,
                )

            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except Exception as api_err:
                last_error = f"API error: {api_err}"
                if "context_length_exceeded" in str(api_err):
                    raise RuntimeError(
                        f"Trajectory {t_idx} exceeds the model's context window. "
                        f"Use a model with a larger context limit or split the trajectory file.\n"
                        f"{api_err}"
                    )
                if verbose:
                    print(f"[LLM-IR]   API error: {api_err}", flush=True)
                continue
            elapsed = round(time.perf_counter() - t0, 2)

            raw_text = (resp.choices[0].message.content or "").strip()
            if verbose:
                print(
                    f"[LLM-IR]   got {len(raw_text)} chars in {elapsed}s",
                    flush=True,
                )

            # --- Parse JSON ---------------------------------------------------
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as je:
                last_error = f"JSON parse error: {je}"
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response was not valid JSON. Error:\n{je}\n\n"
                        "Please fix and return ONLY the corrected JSON."
                    ),
                })
                continue

            # Handle array (multiple trajectories in one response)
            if isinstance(parsed, list):
                ir_candidates = parsed
            else:
                ir_candidates = [parsed]

            # --- Validate each IR object --------------------------------------
            all_valid = True
            validation_errors = []
            for ci, candidate in enumerate(ir_candidates):
                try:
                    validate_ir(candidate)
                except (ValueError, TypeError, KeyError) as ve:
                    all_valid = False
                    validation_errors.append(f"IR[{ci}]: {ve}")

            if all_valid:
                ir_obj = ir_candidates
                break
            else:
                err_summary = "\n".join(validation_errors)
                last_error = err_summary
                if verbose:
                    print(f"[LLM-IR]   validation failed:\n{err_summary}", flush=True)
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"The IR you returned failed validation:\n{err_summary}\n\n"
                        "Please fix the issues and return ONLY the corrected JSON."
                    ),
                })

        if ir_obj is None:
            raise RuntimeError(
                f"LLM IR conversion failed for trajectory {t_idx} after "
                f"{max_retries} attempts. Last error: {last_error}"
            )

        results.extend(ir_obj)

    return results