"""
On-demand tool parameter schema for the workflow generator.

The planner only ever sees the lightweight tool *catalogue* (name + one-line
description — see ``tool_catalogue.py``). That is enough to pick which tool a
``tool`` node should call, but not enough to fill the node's ``tool_parameters``
correctly: the builder needs each tool's real parameter names, types, required
flags and options. Sending every installed tool's full parameter schema to the
planner would blow the context window, so we resolve a tool's detailed schema
*progressively* — only after the planner has committed to a concrete
``provider`` / ``tool`` — via the ``ToolSchemaResolver`` callback injected into
the runner.

Boundary: this module lives on the service side (it touches ``ToolManager`` /
``BuiltinToolManageService`` and the plugin daemon). The runner stays pure and
only ever calls the injected ``ToolSchemaResolver`` callable — it never imports
this module. Mirrors the ``tool_catalogue`` injection pattern.
"""

import logging
from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

from core.tools.entities.tool_entities import ToolParameter
from core.workflow.generator.tool_catalogue import ToolCatalogueEntry

logger = logging.getLogger(__name__)


class ToolParamInfo(TypedDict):
    """One tool parameter, trimmed to what a node builder needs to fill it."""

    name: str
    type: str  # ToolParameterType value: "string" | "number" | "select" | "file" | ...
    required: bool
    # "form"  → the builder must supply a value in the node's tool_parameters.
    # "llm"   → filled at run time by the agent/LLM; NOT part of the static node config.
    form: str
    description: str
    default: NotRequired[Any]
    options: NotRequired[list[str]]  # allowed values for select / dynamic-select


class ToolSchemaInfo(TypedDict):
    """A single tool's resolved schema, LLM-friendly and daemon-decoupled."""

    provider_id: str
    provider_type: str  # "plugin" | "builtin" — what the workflow tool node needs
    tool_name: str
    tool_label: str
    parameters: list[ToolParamInfo]
    output_schema: dict[str, Any]


# ``resolver(provider, tool) -> ToolSchemaInfo | None``. ``None`` means the tool
# couldn't be resolved (unknown pair, daemon error) — the builder then falls
# back to the generic tool template, exactly like the pre-schema behaviour.
ToolSchemaResolver = Callable[[str, str], "ToolSchemaInfo | None"]


def _param_description(param: ToolParameter) -> str:
    """Prefer the LLM-facing description, then the human one, else empty."""
    if param.llm_description:
        return param.llm_description.strip()
    human = getattr(param, "human_description", None)
    if human is not None:
        text = getattr(human, "en_US", "") or getattr(human, "zh_Hans", "")
        if text:
            return text.strip()
    return ""


def _trim_parameter(param: ToolParameter) -> ToolParamInfo | None:
    """Reduce one ToolParameter to the fields a builder needs.

    ``schema``-form parameters are metadata set at install time (not something
    the builder fills), so they are dropped. ``form`` and ``llm`` parameters are
    kept — ``form`` ones must be filled, ``llm`` ones are flagged so the builder
    knows to leave them to the run time.
    """
    form = param.form
    if form == ToolParameter.ToolParameterForm.SCHEMA:
        return None

    info: ToolParamInfo = {
        "name": param.name,
        "type": param.type.value,
        "required": bool(param.required),
        "form": form.value,
        "description": _param_description(param),
    }
    if param.default is not None:
        info["default"] = param.default
    options = [option.value for option in (param.options or []) if getattr(option, "value", None) is not None]
    if options:
        info["options"] = options
    return info


def build_tool_schema_resolver(
    tenant_id: str,
    entries: list[ToolCatalogueEntry],
) -> ToolSchemaResolver:
    """Build a ``(provider, tool) -> ToolSchemaInfo | None`` resolver for a tenant.

    ``entries`` is the already-built tool catalogue; it supplies the correct
    ``provider_type`` (``plugin`` vs ``builtin``) for each installed tool so the
    generated node carries the right value — the generic builder template used
    to hard-code ``builtin``, which is wrong for plugin tools.

    Resolution is lazy and memoised per ``(provider, tool)`` within one
    generation so re-asking for the same tool never hits the daemon twice. Any
    failure (unknown pair, daemon unreachable) returns ``None`` and is logged —
    generation must never break because a schema lookup failed.
    """
    # ``(provider_name, tool_name)`` → catalogue entry, for provider_type / label.
    entry_by_key: dict[tuple[str, str], ToolCatalogueEntry] = {
        (entry["provider_name"], entry["tool_name"]): entry for entry in entries
    }
    cache: dict[tuple[str, str], ToolSchemaInfo | None] = {}

    def resolve(provider: str, tool: str) -> ToolSchemaInfo | None:
        provider = (provider or "").strip()
        tool = (tool or "").strip()
        if not provider or not tool:
            return None
        key = (provider, tool)
        if key in cache:
            return cache[key]

        result: ToolSchemaInfo | None = None
        try:
            result = _resolve_uncached(tenant_id, provider, tool, entry_by_key)
        except Exception:
            logger.exception(
                "Workflow generator: failed to resolve tool schema for %s/%s (tenant %s)",
                provider,
                tool,
                tenant_id,
            )
        cache[key] = result
        return result

    return resolve


def _resolve_uncached(
    tenant_id: str,
    provider: str,
    tool: str,
    entry_by_key: dict[tuple[str, str], ToolCatalogueEntry],
) -> ToolSchemaInfo | None:
    # Imported lazily so this module (and the runner that receives the callable)
    # don't pull the tool-management service graph at import time.
    from services.tools.builtin_tools_manage_service import BuiltinToolManageService

    entry = entry_by_key.get((provider, tool))
    provider_type = entry["provider_type"] if entry else "builtin"
    tool_label = entry["tool_label"] if entry else tool

    api_tools = BuiltinToolManageService.list_builtin_tool_provider_tools(tenant_id, provider)
    target = next((t for t in api_tools if getattr(t, "name", None) == tool), None)
    if target is None:
        return None

    parameters: list[ToolParamInfo] = []
    for param in target.parameters or []:
        trimmed = _trim_parameter(param)
        if trimmed is not None:
            parameters.append(trimmed)

    output_schema = dict(getattr(target, "output_schema", None) or {})

    return {
        "provider_id": provider,
        "provider_type": provider_type,
        "tool_name": tool,
        "tool_label": (getattr(target, "label", None) and _label_text(target.label)) or tool_label,
        "parameters": parameters,
        "output_schema": output_schema,
    }


def _label_text(label: Any) -> str:
    """Pull English label text out of an I18nObject-ish value."""
    if label is None:
        return ""
    return getattr(label, "en_US", "") or getattr(label, "zh_Hans", "") or ""
