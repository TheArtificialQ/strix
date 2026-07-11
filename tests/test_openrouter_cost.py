"""Tests for the TEMPORARY OpenRouter streaming-cost workaround.

Covers both halves of the fix:
  - ``strix.config.models._extract_openrouter_billed_cost`` /
    ``_patch_openrouter_streaming_cost_passthrough`` — carry OpenRouter's billed
    ``usage.cost`` through LiteLLM stream assembly.
  - ``strix.report.state.litellm_cost_callback`` — prefer that billed cost, but
    only when the provider is OpenRouter.

These can all be deleted alongside the workaround once the upstream LiteLLM bugs
(https://github.com/BerriAI/litellm/issues/16021, .../issues/11626) are fixed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from strix.config.models import _extract_openrouter_billed_cost
from strix.report.state import _openrouter_streamed_billed_cost, litellm_cost_callback


def _chunk(provider: str | None, cost: object) -> dict:
    hidden = {"custom_llm_provider": provider} if provider is not None else {}
    return {"_hidden_params": hidden, "usage": {"cost": cost}}


class TestExtractOpenRouterBilledCost:
    def test_returns_billed_cost_from_openrouter_chunk(self) -> None:
        chunks = [_chunk("openrouter", None), _chunk("openrouter", 0.4213)]
        assert _extract_openrouter_billed_cost(chunks) == 0.4213

    def test_ignores_non_openrouter_provider(self) -> None:
        chunks = [_chunk("anthropic", 0.4213)]
        assert _extract_openrouter_billed_cost(chunks) is None

    def test_none_when_no_chunks(self) -> None:
        assert _extract_openrouter_billed_cost([]) is None
        assert _extract_openrouter_billed_cost(None) is None

    def test_ignores_zero_or_negative_cost(self) -> None:
        assert _extract_openrouter_billed_cost([_chunk("openrouter", 0)]) is None
        assert _extract_openrouter_billed_cost([_chunk("openrouter", -1.0)]) is None

    def test_ignores_bool_cost(self) -> None:
        assert _extract_openrouter_billed_cost([_chunk("openrouter", True)]) is None

    def test_reads_cost_from_object_style_chunk(self) -> None:
        usage = SimpleNamespace(cost=0.77)
        chunk = SimpleNamespace(_hidden_params={"custom_llm_provider": "openrouter"}, usage=usage)
        assert _extract_openrouter_billed_cost([chunk]) == 0.77


class TestOpenRouterStreamedBilledCost:
    def test_reads_usage_cost_for_openrouter(self) -> None:
        resp = SimpleNamespace(usage=SimpleNamespace(cost=1.25))
        assert _openrouter_streamed_billed_cost({"custom_llm_provider": "openrouter"}, resp) == 1.25

    def test_none_for_other_providers(self) -> None:
        resp = SimpleNamespace(usage=SimpleNamespace(cost=1.25))
        assert _openrouter_streamed_billed_cost({"custom_llm_provider": "anthropic"}, resp) is None

    def test_none_when_no_usage_cost(self) -> None:
        resp = SimpleNamespace(usage=SimpleNamespace(cost=None))
        assert _openrouter_streamed_billed_cost({"custom_llm_provider": "openrouter"}, resp) is None


class TestLitellmCostCallback:
    def test_prefers_openrouter_billed_cost_over_stale_response_cost(self) -> None:
        # Mapped-but-stale model: response_cost is a positive *wrong* number; the
        # billed usage.cost must win.
        kwargs = {"custom_llm_provider": "openrouter", "response_cost": 0.004}
        resp = SimpleNamespace(usage=SimpleNamespace(cost=0.42), _hidden_params={})
        recorded: list[float] = []
        state = SimpleNamespace(record_observed_llm_cost=recorded.append)

        with patch("strix.report.state.get_global_report_state", return_value=state):
            litellm_cost_callback(kwargs, resp)

        assert recorded == [0.42]

    def test_falls_back_to_response_cost_for_non_openrouter(self) -> None:
        kwargs = {"custom_llm_provider": "anthropic", "response_cost": 0.5}
        resp = SimpleNamespace(usage=SimpleNamespace(cost=999.0), _hidden_params={})
        recorded: list[float] = []
        state = SimpleNamespace(record_observed_llm_cost=recorded.append)

        with patch("strix.report.state.get_global_report_state", return_value=state):
            litellm_cost_callback(kwargs, resp)

        # usage.cost is ignored off-OpenRouter; the SDK-reported response_cost wins.
        assert recorded == [0.5]

    def test_openrouter_without_usage_cost_falls_back_to_response_cost(self) -> None:
        kwargs = {"custom_llm_provider": "openrouter", "response_cost": 0.03}
        resp = SimpleNamespace(usage=SimpleNamespace(cost=None), _hidden_params={})
        recorded: list[float] = []
        state = SimpleNamespace(record_observed_llm_cost=recorded.append)

        with patch("strix.report.state.get_global_report_state", return_value=state):
            litellm_cost_callback(kwargs, resp)

        assert recorded == [0.03]
