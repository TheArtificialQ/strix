from typing import Any, Literal

from strix.config import Config
from strix.config.config import resolve_llm_config
from strix.llm.utils import resolve_strix_model


class LLMConfig:
    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        enable_prompt_caching: bool = True,
        skills: list[str] | None = None,
        timeout: int | None = None,
        scan_mode: str = "deep",
        is_whitebox: bool = False,
        interactive: bool = False,
        reasoning_effort: str | None = None,
        system_prompt_context: dict[str, Any] | None = None,
        role: Literal["root", "subagent"] = "root",
    ):
        resolved_model, resolved_api_key, resolved_api_base = resolve_llm_config(
            role=role,
            fallback_model=model_name,
            fallback_api_key=api_key,
            fallback_api_base=api_base,
        )
        self.role = role
        self.model_name = resolved_model
        self.api_key = resolved_api_key
        self.api_base = resolved_api_base

        if not self.model_name:
            raise ValueError("STRIX_LLM environment variable must be set and not empty")

        api_model, canonical = resolve_strix_model(self.model_name)
        self.litellm_model: str = api_model or self.model_name
        self.canonical_model: str = canonical or self.model_name
        (
            self.subagent_model_name,
            self.subagent_api_key,
            self.subagent_api_base,
        ) = resolve_llm_config(
            role="subagent",
            fallback_model=self.model_name,
            fallback_api_key=self.api_key,
            fallback_api_base=self.api_base,
        )

        self.enable_prompt_caching = enable_prompt_caching
        self.skills = skills or []

        self.timeout = timeout or int(Config.get("llm_timeout") or "300")

        self.scan_mode = scan_mode if scan_mode in ["quick", "standard", "deep"] else "deep"
        self.is_whitebox = is_whitebox
        self.interactive = interactive
        self.reasoning_effort = reasoning_effort
        self.system_prompt_context = system_prompt_context or {}
