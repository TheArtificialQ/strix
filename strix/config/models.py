"""SDK model configuration helpers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from agents import set_default_openai_api, set_default_openai_key, set_tracing_disabled
from agents.models.multi_provider import MultiProvider
from agents.retry import (
    ModelRetryBackoffSettings,
    ModelRetrySettings,
    retry_policies,
)


if TYPE_CHECKING:
    from agents.models.interface import ModelProvider

    from strix.config.settings import Settings


class StrixProvider(MultiProvider):
    """Route any non-OpenAI prefix through LiteLLM with the prefix preserved,
    so users type ``deepseek/deepseek-chat`` rather than
    ``litellm/deepseek/deepseek-chat``.
    """

    def _resolve_prefixed_model(
        self,
        *,
        original_model_name: str,
        prefix: str,
        stripped_model_name: str | None,
    ) -> tuple[ModelProvider, str | None]:
        if prefix in {"openai", "litellm", "any-llm"}:
            return super()._resolve_prefixed_model(
                original_model_name=original_model_name,
                prefix=prefix,
                stripped_model_name=stripped_model_name,
            )
        return self._get_fallback_provider("litellm"), original_model_name


DEFAULT_MODEL_RETRY = ModelRetrySettings(
    max_retries=5,
    backoff=ModelRetryBackoffSettings(
        initial_delay=2.0,
        max_delay=90.0,
        multiplier=2.0,
        jitter=False,
    ),
    policy=retry_policies.any(
        retry_policies.provider_suggested(),
        retry_policies.network_error(),
        retry_policies.http_status((429, 500, 502, 503, 504)),
    ),
)


def configure_sdk_model_defaults(settings: Settings) -> None:
    """Apply Strix config to SDK-native defaults."""
    llm = settings.llm
    set_tracing_disabled(True)
    _configure_litellm_compatibility()
    if llm.api_key:
        set_default_openai_key(llm.api_key, use_for_tracing=False)
        _configure_litellm_default("api_key", llm.api_key)
        _mirror_api_key_to_provider_env(llm.model, llm.api_key)
    if llm.api_base:
        os.environ["OPENAI_BASE_URL"] = llm.api_base
        _configure_litellm_default("api_base", llm.api_base)
        set_default_openai_api("chat_completions")
    else:
        set_default_openai_api("responses")


def _mirror_api_key_to_provider_env(model_name: str | None, api_key: str) -> None:
    if not model_name:
        return
    import litellm

    name = model_name.strip()
    for prefix in ("litellm/", "any-llm/"):
        if name.lower().startswith(prefix):
            name = name[len(prefix) :]
            break
    try:
        report = litellm.validate_environment(model=name.lower())
    except Exception:  # noqa: BLE001
        return
    for env_key in report.get("missing_keys") or []:
        if env_key.endswith("_API_KEY"):
            os.environ.setdefault(env_key, api_key)


def _configure_litellm_compatibility() -> None:
    """Enable LiteLLM's permissive param handling and disable its callbacks."""
    import litellm

    litellm.drop_params = True
    litellm.modify_params = True
    litellm.turn_off_message_logging = True
    litellm.disable_streaming_logging = True
    litellm.suppress_debug_info = True

    _register_litellm_cost_callback()
    _patch_openrouter_streaming_cost_passthrough()


# ---------------------------------------------------------------------------
# TEMPORARY OpenRouter-only workaround. Remove once the upstream LiteLLM bugs
# are fixed and we bump past the pinned version.
#
# OpenRouter returns the exact billed cost in the final stream chunk's
# ``usage.cost`` field, and LiteLLM's OpenRouter integration explicitly asks
# for it (it forces ``usage: {include: true}`` on every request). On the
# NON-streaming path LiteLLM propagates that value correctly. On the STREAMING
# path — the only path Strix uses — the generic stream assembler
# (``ChunkProcessor.calculate_usage``) rebuilds the ``Usage`` object field by
# field and never copies ``cost``, so the real billed amount is dropped and
# LiteLLM falls back to its static price map. That map is missing or stale for
# most ``openrouter/*`` slugs, which is why reported cost is ~100x off (or $0).
#
# Upstream issues:
#   - https://github.com/BerriAI/litellm/issues/16021
#   - https://github.com/BerriAI/litellm/issues/11626
#
# This patch carries OpenRouter's billed ``usage.cost`` through stream
# assembly so ``litellm_cost_callback`` can read it. It is intentionally
# scoped to OpenRouter chunks only and is a no-op for every other provider.
# ---------------------------------------------------------------------------
def _patch_openrouter_streaming_cost_passthrough() -> None:
    try:
        from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
    except Exception:  # noqa: BLE001 - never let a compatibility shim break startup
        return

    if getattr(ChunkProcessor, "_strix_openrouter_cost_passthrough", False):
        return

    original_calculate_usage = ChunkProcessor.calculate_usage

    def calculate_usage_with_openrouter_cost(  # type: ignore[no-untyped-def]
        self: Any, *args: Any, **kwargs: Any
    ) -> Any:
        usage = original_calculate_usage(self, *args, **kwargs)
        try:
            chunks = kwargs.get("chunks") if "chunks" in kwargs else (args[0] if args else None)
            billed_cost = _extract_openrouter_billed_cost(chunks)
            if billed_cost is not None:
                usage.cost = billed_cost
        except Exception:  # noqa: BLE001 - preserve stock behavior on any surprise
            return usage
        return usage

    ChunkProcessor.calculate_usage = calculate_usage_with_openrouter_cost  # type: ignore[method-assign]
    ChunkProcessor._strix_openrouter_cost_passthrough = True  # type: ignore[attr-defined]


def _extract_openrouter_billed_cost(chunks: Any) -> float | None:
    """Return OpenRouter's billed ``usage.cost`` from stream chunks, else ``None``.

    Only returns a value when the chunks are from OpenRouter (per each chunk's
    ``_hidden_params['custom_llm_provider']``) and a positive numeric cost is
    present, so non-OpenRouter providers are never affected.
    """
    if not chunks:
        return None

    for chunk in reversed(list(chunks)):
        if _chunk_provider(chunk) != "openrouter":
            return None

        chunk_usage = (
            chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
        )
        if chunk_usage is None:
            continue
        cost = (
            chunk_usage.get("cost")
            if isinstance(chunk_usage, dict)
            else getattr(chunk_usage, "cost", None)
        )
        if isinstance(cost, bool):
            continue
        if isinstance(cost, int | float) and cost > 0:
            return float(cost)

    return None


def _chunk_provider(chunk: Any) -> str | None:
    hidden_params = (
        chunk.get("_hidden_params")
        if isinstance(chunk, dict)
        else getattr(chunk, "_hidden_params", None)
    )
    if isinstance(hidden_params, dict):
        provider = hidden_params.get("custom_llm_provider")
        if isinstance(provider, str):
            return provider
    return None


def _register_litellm_cost_callback() -> None:
    import litellm

    from strix.report.state import litellm_cost_callback

    for bucket_name in ("success_callback", "_async_success_callback"):
        bucket = getattr(litellm, bucket_name, None)
        if not isinstance(bucket, list):
            continue
        if litellm_cost_callback in bucket:
            continue
        bucket.append(litellm_cost_callback)


def _configure_litellm_default(name: str, value: str) -> None:
    """Set LiteLLM's module-level defaults without adding a provider wrapper."""
    import litellm

    setattr(litellm, name, value)


def uses_chat_completions_tool_schema(model_name: str, settings: Settings) -> bool:
    """Return whether the resolved SDK route can only receive JSON function tools."""
    model = model_name.strip().lower()
    if "/" in model and not model.startswith("openai/"):
        return True
    if settings.llm.api_base:
        return True
    return not model_supports_reasoning(model_name)


def model_supports_reasoning(model_name: str) -> bool:
    import litellm

    name = model_name.strip().lower()
    for prefix in ("litellm/", "any-llm/", "openai/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    entry = litellm.model_cost.get(name)
    if entry is None and "/" in name:
        entry = litellm.model_cost.get(name.rsplit("/", 1)[1])
    return bool(entry and entry.get("supports_reasoning"))


def is_known_openai_bare_model(model_name: str) -> bool:
    import litellm

    name = model_name.strip().lower()
    if not name or "/" in name:
        return False
    entry = litellm.model_cost.get(name)
    return bool(entry and entry.get("litellm_provider") == "openai")
