"""Tests for copying user-provided input files into report output dirs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strix.report.state import ReportState
from strix.report.writer import write_input_artifacts, write_report_readme


if TYPE_CHECKING:
    import pytest


def test_write_input_artifacts_copies_configured_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    instruction_file = source_dir / "scope.md"
    setup_script = source_dir / "prepare.sh"
    instruction_file.write_text("Focus on auth flows.\n", encoding="utf-8")
    setup_script.write_text("#!/usr/bin/env bash\necho ready\n", encoding="utf-8")
    run_dir = tmp_path / "report"
    run_record: dict[str, Any] = {
        "instruction_file": str(instruction_file),
        "setup_script": str(setup_script),
    }

    artifacts = write_input_artifacts(run_dir, run_record)

    assert artifacts == {
        "instruction_file": "instruction-file-scope.md",
        "setup_script": "setup-script-prepare.sh",
    }
    assert (run_dir / "instruction-file-scope.md").read_text(encoding="utf-8") == (
        "Focus on auth flows.\n"
    )
    assert (run_dir / "setup-script-prepare.sh").read_text(encoding="utf-8") == (
        "#!/usr/bin/env bash\necho ready\n"
    )
    assert run_record["input_artifacts"] == artifacts


def test_write_input_artifacts_keeps_existing_copy_when_refresh_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    instruction_file = source_dir / "scope.md"
    instruction_file.write_text("new copy\n", encoding="utf-8")
    run_dir = tmp_path / "report"
    run_dir.mkdir()
    target_path = run_dir / "instruction-file-scope.md"
    target_path.write_text("previous saved copy\n", encoding="utf-8")
    run_record: dict[str, Any] = {
        "instruction_file": str(instruction_file),
        "input_artifacts": {"instruction_file": target_path.name},
    }

    def fail_copy2(_source_path: Path, tmp_path: Path) -> None:
        tmp_path.write_text("partial copy\n", encoding="utf-8")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("strix.report.writer.shutil.copy2", fail_copy2)

    artifacts = write_input_artifacts(run_dir, run_record)

    assert artifacts == {"instruction_file": "instruction-file-scope.md"}
    assert target_path.read_text(encoding="utf-8") == "previous saved copy\n"
    assert list(run_dir.glob(".instruction-file-scope.md.*.tmp")) == []


def test_write_report_readme_removes_temp_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "report"

    def fail_replace(_source_path: Path, _target_path: Path) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "replace", fail_replace)

    try:
        write_report_readme(run_dir, {})
    except OSError:
        pass
    else:  # pragma: no cover - defensive assertion clarity
        raise AssertionError("write_report_readme should have raised OSError")

    assert list(run_dir.glob(".README.md.*.tmp")) == []


def test_write_report_readme_renders_run_notes(tmp_path: Path) -> None:
    run_dir = tmp_path / "sample-run"
    state_dir = run_dir / ".state"
    state_dir.mkdir(parents=True)
    with sqlite3.connect(state_dir / "agents.db") as conn:
        conn.execute(
            "create table agent_messages ("
            "id integer primary key, session_id text, message_data text, created_at text)"
        )
        conn.executemany(
            "insert into agent_messages (session_id, message_data, created_at) values (?, ?, ?)",
            [
                (
                    "root",
                    json.dumps({"type": "function_call", "call_id": "call-1"}),
                    "2026-06-25T10:01:00+00:00",
                ),
                (
                    "root",
                    json.dumps({"type": "function_call_output", "call_id": "call-1"}),
                    "2026-06-25T10:02:00+00:00",
                ),
                (
                    "child",
                    json.dumps({"type": "function_call", "call_id": "call-2"}),
                    "2026-06-25T10:03:00+00:00",
                ),
            ],
        )

    write_report_readme(
        run_dir,
        {
            "targets_info": [{"original": "https://app.example"}],
            "llm_model": "openai/gpt-5.4",
            "app_version": "1.2.3",
            "start_time": "2026-06-25T10:00:00+00:00",
            "end_time": "2026-06-25T10:12:36+00:00",
            "llm_usage": {"cost": 4.567, "total_tokens": 1_234_567},
        },
    )

    readme = (run_dir / "README.md").read_text(encoding="utf-8")
    assert "Target: https://app.example\n" in readme
    assert "LLM Model: openai/gpt-5.4\n" in readme
    assert "App Version: 1.2.3\n" in readme
    assert "| sample-run |" in readme
    assert "| 0.0 | 4.57 | 12.6 | 1.2 | 2 |" in readme


def test_report_state_saves_input_artifacts_from_scan_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    instruction_file = tmp_path / "instructions.txt"
    setup_script = tmp_path / "setup.sh"
    instruction_file.write_text("Use tenant test accounts.\n", encoding="utf-8")
    setup_script.write_text("#!/usr/bin/env bash\necho seeded\n", encoding="utf-8")

    report_state = ReportState("artifact-run")
    report_state.set_scan_config(
        {
            "targets": [],
            "user_instructions": "Use tenant test accounts.",
            "scan_mode": "deep",
            "diff_scope": {"active": False},
            "non_interactive": True,
            "local_sources": [],
            "scope_mode": "auto",
            "diff_base": None,
            "instruction_file": str(instruction_file),
            "setup_script": str(setup_script),
            "docker_network": None,
            "llm_model": "openai/gpt-5.4",
        }
    )

    report_state.save_run_data()

    run_dir = tmp_path / "strix_runs" / "artifact-run"
    assert (run_dir / "README.md").exists()
    assert (run_dir / "instruction-file-instructions.txt").read_text(
        encoding="utf-8"
    ) == "Use tenant test accounts.\n"
    assert (run_dir / "setup-script-setup.sh").read_text(encoding="utf-8") == (
        "#!/usr/bin/env bash\necho seeded\n"
    )
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_record["input_artifacts"] == {
        "instruction_file": "instruction-file-instructions.txt",
        "setup_script": "setup-script-setup.sh",
    }
