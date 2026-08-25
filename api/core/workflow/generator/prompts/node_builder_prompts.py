"""Compact prompts for parallel, per-node workflow configuration.

Each call produces only the semantic ``data`` fields for one planned node.
Canvas wrappers, shared labels, topology, layout, and edge defaults are owned
by ``WorkflowGenerator`` so completion length scales with node configuration
rather than with the full ReactFlow graph.
"""

import json
from typing import Any

from core.workflow.generator.prompts.builder_prompts import get_node_config_snippet

_CONTAINER_CONFIG_SNIPPETS = {
    "iteration": """- iteration:
    {"iterator_selector": ["<src>", "<list-var>"],
     "output_selector": ["<last-child>", "<out-var>"],
     "is_parallel": false, "parallel_nums": 10,
     "error_handle_mode": "terminated", "flatten_output": true}
    The runner supplies start_node_id, child wrappers, and the synthetic start node.""",
    "loop": """- loop:
    {"break_conditions": [{"id": "c1",
                            "variable_selector": ["<child>", "<var>"],
                            "comparison_operator": "is",
                            "value": "<value>"}],
     "loop_count": 10, "logical_operator": "and"}
    The runner supplies start_node_id, child wrappers, and the synthetic start node.""",
}

_NODE_BUILDER_HEAD = """You configure exactly ONE node in a Dify workflow.

Return one JSON object with exactly this shape: {"config": {...}}.
``config`` contains only node-type-specific ``data`` fields. Do NOT repeat id,
type, title, desc, selected, position, wrapper fields, edges, or viewport.

Rules:
- Use only ids from the supplied normalized plan.
- Placeholder strings use ``{{#node_id.variable#}}``; selector fields use
  ``["node_id", "variable"]``. Never invent an upstream output.
- Use the selected model verbatim for llm, question-classifier, and
  parameter-extractor nodes.
- Keep prompts/code concise but complete for the user's requested behavior.
- Emit strict JSON only: no prose, Markdown, comments, or trailing commas.

# Target node schema

"""


NODE_BUILDER_USER_PROMPT = """# Target node

id={node_id}, type={node_type}, label={label!r}
purpose={purpose}

# User instruction

{instruction}

{ideal_output_section}{mode_section}{model_section}{tool_catalogue_section}{start_inputs_section}{variable_contract_section}{existing_config_section}\
# Normalized plan and topology

{plan_json}

Return {{"config": {{...}}}} for target node {node_id} now.
"""


def get_node_builder_system_prompt(node_type: str) -> str:
    """Build a one-node prompt containing only that node's semantic schema."""
    snippet = _CONTAINER_CONFIG_SNIPPETS.get(node_type) or get_node_config_snippet(node_type)
    return _NODE_BUILDER_HEAD + (snippet or f"- {node_type}: emit the minimum valid config fields.")


def format_parallel_plan(
    plan_nodes: list[dict[str, Any]],
    plan_edges: list[dict[str, Any]],
    start_inputs: list[dict[str, Any]] | None = None,
) -> str:
    """Serialize the shared plan compactly so every node call has graph context.

    ``start_inputs`` rides along so downstream builders reference the declared
    ``{{#<start-id>.<variable>#}}`` names instead of guessing them from prose
    — a guessed name gets auto-injected as a spurious form input later.
    """
    payload: dict[str, Any] = {"nodes": plan_nodes, "edges": plan_edges}
    if start_inputs:
        payload["start_inputs"] = start_inputs
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def format_mode_section(mode: str) -> str:
    """Tell each builder which app mode it is configuring for.

    Matters most in advanced-chat, where ``sys.query`` / ``sys.files`` are the
    sanctioned way to reference the user's message — without this the model
    invents start-node variables that postprocess then materializes as
    spurious form inputs.
    """
    if mode == "advanced-chat":
        return (
            "# App mode\n\n"
            "advanced-chat: the user's chat message is available as sys.query and uploaded files "
            'as sys.files — placeholder {{#sys.query#}}, selector ["sys", "query"]. Reference them '
            "directly; do NOT invent start-node variables for the chat message.\n\n"
        )
    return (
        "# App mode\n\n"
        "workflow: there are NO automatic system variables; reference user input only through "
        "the start node's declared variables.\n\n"
    )


def format_start_inputs_section(start_inputs: list[dict[str, Any]]) -> str:
    """Render planner-declared inputs for the start-node builder only."""
    if not start_inputs:
        return ""
    lines = ["# Start inputs (copy each entry verbatim into start.data.variables)", ""]
    for input_ in start_inputs:
        variable = str(input_.get("variable") or "").strip()
        if not variable:
            continue
        label = str(input_.get("label") or "").strip()
        type_ = str(input_.get("type") or "paragraph").strip()
        lines.append(f"- variable={variable!r}  label={label!r}  type={type_!r}")
    lines.append("")
    return "\n".join(lines) + "\n"


def format_variable_contract_section(inputs: "list[Any] | None", outputs: "list[Any] | None") -> str:
    """Render the planner-declared variable wiring for one target node.

    The planner (rule 16) declares, per node, which upstream values it consumes
    (``inputs``: ``[[src_id, var], ...]``) and — for custom-output nodes — which
    variables it exposes (``outputs``). Surfacing this to the builder makes the
    parallel builders agree on names instead of each guessing an upstream output
    (the dominant source of unresolved ``{{#node.var#}}`` references). Returns ""
    when the planner declared nothing (older prompts), so behaviour degrades to
    the previous guess-and-repair path.
    """
    lines: list[str] = []
    ref_pairs = [
        p for p in (inputs or [])
        if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, str) and x.strip() for x in p)
    ]
    if ref_pairs:
        lines.append("# Variable contract — reference ONLY these upstream outputs")
        lines.append("")
        lines.append(
            "When you reference an upstream value (placeholder {{#src.var#}} or "
            'selector ["src", "var"]) use EXACTLY these source ids and variable '
            "names. Do NOT invent other names:"
        )
        for src, var in ref_pairs:
            lines.append(f'- {{{{#{src}.{var}#}}}}   (selector: [{src!r}, {var!r}])')
        lines.append("")
    out_names = [str(o) for o in (outputs or []) if isinstance(o, str) and o.strip()]
    if out_names:
        lines.append(
            "This node MUST expose exactly these output variable names so downstream "
            f"nodes resolve: {', '.join(out_names)}. Define them in the config accordingly."
        )
        lines.append("")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def format_tool_catalogue_section(catalogue_text: str) -> str:
    """Render exact tool identifiers for a tool-node builder only."""
    if not catalogue_text.strip():
        return ""
    return (
        "# Available tools (use these exact provider/tool identifiers — "
        "set provider_id and provider_name to the provider portion and "
        "tool_name to the tool portion)\n\n"
        f"{catalogue_text}\n\n"
    )


def format_tool_schema_section(schema: "dict[str, Any] | None") -> str:
    """Render one resolved tool's real parameter schema for its node builder.

    Progressive loading: the planner picked a concrete ``provider`` / ``tool``
    from the lightweight catalogue; here — only for that chosen tool — we inject
    its actual parameter list (names, types, required flags, options) so the
    builder fills ``tool_parameters`` against the real contract instead of
    guessing. Returns "" when the schema couldn't be resolved, in which case the
    builder falls back to the generic ``tool`` template in ``_NODE_SNIPPETS``.

    ``schema`` is a ``ToolSchemaInfo`` dict (see
    ``core.workflow.generator.tool_schema``). Only ``form == "form"`` parameters
    need a value in the node; ``form == "llm"`` ones are listed so the builder
    knows to leave them for the run time.
    """
    if not schema:
        return ""

    provider_id = str(schema.get("provider_id") or "")
    provider_type = str(schema.get("provider_type") or "builtin")
    tool_name = str(schema.get("tool_name") or "")
    tool_label = str(schema.get("tool_label") or tool_name)
    parameters = schema.get("parameters") or []

    lines = [
        "# Selected tool schema (fill tool_parameters against THIS contract)",
        "",
        f"provider_id={provider_id!r}  provider_type={provider_type!r}  "
        f"tool_name={tool_name!r}  tool_label={tool_label!r}",
        "",
        "Set the node's provider_id, provider_name, provider_type, tool_name and "
        "tool_label to exactly these values. provider_type MUST be the value above "
        f"({provider_type!r}) — do not hard-code 'builtin' for a plugin tool.",
        "",
    ]

    form_params = [p for p in parameters if p.get("form") == "form"]
    llm_params = [p for p in parameters if p.get("form") == "llm"]

    if form_params:
        lines.append("Parameters you MUST provide in tool_parameters (form params):")
        for p in form_params:
            lines.append(_render_param_line(p))
        lines.append("")
        lines.append(
            "For each: use {\"type\":\"mixed\",\"value\":\"...{{#src.var#}}...\"} for a "
            "string template, {\"type\":\"variable\",\"value\":[\"src\",\"var\"]} for a "
            "direct reference, or {\"type\":\"constant\",\"value\":<literal>} for a fixed "
            "value. Only include a required param you cannot fill from an upstream "
            "output as a constant."
        )
        lines.append("")
    else:
        lines.append("This tool has no form parameters to fill; tool_parameters may be {}.")
        lines.append("")

    if llm_params:
        names = ", ".join(str(p.get("name")) for p in llm_params)
        lines.append(f"Run-time (LLM-filled) params — do NOT put these in tool_parameters: {names}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_param_line(param: "dict[str, Any]") -> str:
    """One bullet describing a single form parameter for the builder."""
    name = str(param.get("name") or "")
    ptype = str(param.get("type") or "")
    required = "required" if param.get("required") else "optional"
    parts = [f"- {name} ({ptype}, {required})"]
    desc = str(param.get("description") or "").strip()
    if desc:
        parts.append(f": {desc}")
    if param.get("options"):
        opts = ", ".join(str(o) for o in param["options"])
        parts.append(f" [options: {opts}]")
    if "default" in param:
        parts.append(f" [default: {param['default']!r}]")
    return "".join(parts)
