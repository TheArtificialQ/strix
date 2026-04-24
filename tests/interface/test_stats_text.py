import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_utils_module():
    module_path = Path(__file__).resolve().parents[2] / "strix" / "interface" / "utils.py"
    spec = importlib.util.spec_from_file_location("strix_interface_utils_stats_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load strix.interface.utils for tests")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_utils_module()


class DummyTracer:
    def __init__(self, llm_stats, caido_url: str | None = None) -> None:
        self._llm_stats = llm_stats
        self.caido_url = caido_url

    def get_total_llm_stats(self):
        return self._llm_stats


def test_build_tui_stats_text_includes_per_model_breakdown_when_multiple_models_used() -> None:
    tracer = DummyTracer(
        {
            "total": {
                "input_tokens": 3_000,
                "output_tokens": 750,
                "cached_tokens": 500,
                "cost": 0.6667,
                "requests": 5,
            },
            "total_tokens": 3_750,
            "by_model": {
                "openai/gpt-4.1-mini": {
                    "input_tokens": 2_000,
                    "output_tokens": 500,
                    "cached_tokens": 400,
                    "cost": 0.5432,
                    "requests": 3,
                    "total_tokens": 2_500,
                },
                "openai/gpt-5.4": {
                    "input_tokens": 1_000,
                    "output_tokens": 250,
                    "cached_tokens": 100,
                    "cost": 0.1235,
                    "requests": 2,
                    "total_tokens": 1_250,
                },
            },
        }
    )

    text = utils.build_tui_stats_text(
        tracer,
        {
            "llm_config": SimpleNamespace(
                model_name="openai/gpt-5.4",
                subagent_model_name="openai/gpt-4.1-mini",
            )
        },
    )

    lines = text.plain.splitlines()

    assert lines == [
        "openai/gpt-5.4",
        "Subagents: openai/gpt-4.1-mini",
        "3.8K tokens · $0.67",
        "By Model",
        "openai/gpt-4.1-mini: 2.5K tokens · $0.54",
        "openai/gpt-5.4: 1.2K tokens · $0.12",
    ]


def test_build_tui_stats_text_skips_breakdown_until_multiple_models_have_usage() -> None:
    tracer = DummyTracer(
        {
            "total": {
                "input_tokens": 1_000,
                "output_tokens": 250,
                "cached_tokens": 100,
                "cost": 0.1235,
                "requests": 2,
            },
            "total_tokens": 1_250,
            "by_model": {
                "openai/gpt-5.4": {
                    "input_tokens": 1_000,
                    "output_tokens": 250,
                    "cached_tokens": 100,
                    "cost": 0.1235,
                    "requests": 2,
                    "total_tokens": 1_250,
                }
            },
        }
    )

    text = utils.build_tui_stats_text(
        tracer,
        {
            "llm_config": SimpleNamespace(
                model_name="openai/gpt-5.4",
                subagent_model_name="openai/gpt-4.1-mini",
            )
        },
    )

    assert "By Model" not in text.plain