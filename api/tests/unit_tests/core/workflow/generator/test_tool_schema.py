"""Unit tests for the progressive tool-schema resolver and its prompt section."""

from types import SimpleNamespace
from unittest.mock import patch

from core.workflow.generator.prompts.node_builder_prompts import format_tool_schema_section
from core.workflow.generator.tool_catalogue import ToolCatalogueEntry
from core.workflow.generator.tool_schema import (
    _trim_parameter,
    build_tool_schema_resolver,
)


def _entry(provider: str, tool: str, *, provider_type: str = "plugin", label: str = "") -> ToolCatalogueEntry:
    return ToolCatalogueEntry(
        provider_name=provider,
        provider_type=provider_type,
        plugin_id="langgenius" if provider_type == "plugin" else "",
        tool_name=tool,
        tool_label=label,
        description="",
    )


def _param(name, *, ptype="string", required=False, form="form", options=None, default=None, llm_desc=""):
    """Build a ToolParameter-like stub with only the attributes the trimmer reads."""
    form_enum = SimpleNamespace(value=form)
    type_enum = SimpleNamespace(value=ptype)
    option_objs = [SimpleNamespace(value=o) for o in (options or [])]
    return SimpleNamespace(
        name=name,
        type=type_enum,
        required=required,
        form=form_enum,
        options=option_objs,
        default=default,
        llm_description=llm_desc,
        human_description=None,
    )


class _FakeForm:
    """Mimics ToolParameter.ToolParameterForm.SCHEMA identity check."""

    SCHEMA = SimpleNamespace(value="schema")


class TestTrimParameter:
    def test_schema_form_param_is_dropped(self, monkeypatch):
        # A ``schema``-form param is install-time metadata, not something the
        # builder fills, so it must be excluded from the LLM-facing spec.
        import core.workflow.generator.tool_schema as mod

        schema_param = _param("meta", form="schema")
        monkeypatch.setattr(mod, "ToolParameter", SimpleNamespace(ToolParameterForm=_FakeForm))
        assert _trim_parameter(schema_param) is None

    def test_form_param_is_trimmed_to_needed_fields(self, monkeypatch):
        import core.workflow.generator.tool_schema as mod

        monkeypatch.setattr(mod, "ToolParameter", SimpleNamespace(ToolParameterForm=_FakeForm))
        info = _trim_parameter(
            _param("query", ptype="string", required=True, form="form", llm_desc="the search query")
        )
        assert info == {
            "name": "query",
            "type": "string",
            "required": True,
            "form": "form",
            "description": "the search query",
        }

    def test_options_and_default_included_when_present(self, monkeypatch):
        import core.workflow.generator.tool_schema as mod

        monkeypatch.setattr(mod, "ToolParameter", SimpleNamespace(ToolParameterForm=_FakeForm))
        info = _trim_parameter(
            _param("kind", ptype="select", form="form", options=["a", "b"], default="a")
        )
        assert info["options"] == ["a", "b"]
        assert info["default"] == "a"


class TestBuildToolSchemaResolver:
    def test_returns_none_for_blank_provider_or_tool(self):
        resolver = build_tool_schema_resolver("t1", [])
        assert resolver("", "x") is None
        assert resolver("x", "") is None

    def test_daemon_failure_degrades_to_none(self):
        # A lookup failure must never break generation — the builder then falls
        # back to the generic catalogue template.
        resolver = build_tool_schema_resolver("t1", [_entry("langgenius/google", "google_search")])
        with patch(
            "services.tools.builtin_tools_manage_service.BuiltinToolManageService.list_builtin_tool_provider_tools",
            side_effect=RuntimeError("daemon down"),
        ):
            assert resolver("langgenius/google", "google_search") is None

    def test_resolves_and_memoises(self):
        entries = [_entry("langgenius/google", "google_search", provider_type="plugin", label="Google Search")]
        resolver = build_tool_schema_resolver("t1", entries)
        api_tool = SimpleNamespace(
            name="google_search",
            label=SimpleNamespace(en_US="Google Search"),
            parameters=[_param("query", required=True), _param("secret", form="llm")],
            output_schema={"type": "object"},
        )

        # ToolParameterForm identity needs to match real enum for _trim; patch it.
        import core.workflow.generator.tool_schema as mod

        with patch(
            "services.tools.builtin_tools_manage_service.BuiltinToolManageService.list_builtin_tool_provider_tools",
            return_value=[api_tool],
        ) as mocked, patch.object(mod, "ToolParameter", SimpleNamespace(ToolParameterForm=_FakeForm)):
            first = resolver("langgenius/google", "google_search")
            second = resolver("langgenius/google", "google_search")

        assert first is not None
        assert first["provider_type"] == "plugin"  # from catalogue entry, not hard-coded builtin
        assert first["tool_label"] == "Google Search"
        assert [p["name"] for p in first["parameters"]] == ["query", "secret"]
        assert first["output_schema"] == {"type": "object"}
        # Memoised: a second lookup for the same pair must not hit the service again.
        assert mocked.call_count == 1
        assert second is first


class TestFormatToolSchemaSection:
    def test_none_returns_empty(self):
        assert format_tool_schema_section(None) == ""

    def test_renders_form_params_and_flags_llm_params(self):
        schema = {
            "provider_id": "langgenius/google",
            "provider_type": "plugin",
            "tool_name": "google_search",
            "tool_label": "Google Search",
            "parameters": [
                {"name": "query", "type": "string", "required": True, "form": "form", "description": "the query"},
                {"name": "api_key", "type": "secret-input", "required": True, "form": "llm", "description": ""},
            ],
            "output_schema": {},
        }
        text = format_tool_schema_section(schema)
        assert "provider_type='plugin'" in text
        assert "- query (string, required): the query" in text
        # LLM-filled params must be flagged as NOT going into tool_parameters.
        assert "do NOT put these in tool_parameters: api_key" in text

    def test_no_form_params_says_empty_allowed(self):
        schema = {
            "provider_id": "p",
            "provider_type": "builtin",
            "tool_name": "t",
            "tool_label": "T",
            "parameters": [],
            "output_schema": {},
        }
        text = format_tool_schema_section(schema)
        assert "tool_parameters may be {}" in text
