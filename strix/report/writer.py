"""Artifact writers for Strix scan reports."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from strix.core.paths import run_record_path, runtime_state_dir


logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
_INPUT_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("instruction_file", "instruction-file"),
    ("setup_script", "setup-script"),
)
_README_RESULTS_HEADER = (
    "| Test | VB-4.8 | VB-5.3A | VB-5.3B | VB-5.3C | VB-5.4 | "
    "VB-7.5 | VB-7.6 | VB-8.3 | VB-8.6 | VB-8.7 | VB-9.3 | "
    "VB-9.4 | VB-9.8 | VB-9.9 | Score | Cost ($) | Time (min) | "
    "Tokens (M) | Tools |"
)
_README_RESULTS_ALIGNMENT = (
    "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ":---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|"
)


def read_run_record(run_dir: Path) -> dict[str, Any]:
    path = run_record_path(run_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"run.json at {path} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"run.json at {path} is not an object")
    return data


def write_run_record(run_dir: Path, run_record: dict[str, Any]) -> None:
    _atomic_write_text(
        run_record_path(run_dir),
        json.dumps(run_record, ensure_ascii=False, indent=2, default=str),
    )


def write_executive_report(run_dir: Path, final_scan_result: str) -> None:
    path = run_dir / "penetration_test_report.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Security Penetration Test Report\n\n")
        f.write(f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"{final_scan_result}\n")
    logger.info("Saved final penetration test report to: %s", path)


def write_report_readme(run_dir: Path, run_record: dict[str, Any]) -> None:
    path = run_dir / "README.md"
    _atomic_write_text(path, render_report_readme(run_dir, run_record))
    logger.info("Saved test run notes to: %s", path)


def render_report_readme(run_dir: Path, run_record: dict[str, Any]) -> str:
    now = datetime.now(UTC)
    usage = run_record.get("llm_usage") if isinstance(run_record.get("llm_usage"), dict) else {}
    cost = _float_or_zero(usage.get("cost") if isinstance(usage, dict) else None)
    tokens_m = _total_tokens(usage) / 1_000_000
    elapsed_min = _elapsed_minutes(run_record, now)
    tool_calls = _count_tool_calls(run_dir)

    report_name = _table_cell(run_dir.name)
    row = _markdown_row(
        [
            report_name,
            *([""] * 14),
            "0.0",
            f"{cost:.2f}",
            f"{elapsed_min:.1f}",
            f"{tokens_m:.1f}",
            str(tool_calls),
        ]
    )

    return "\n".join(
        [
            "# Test run notes",
            "",
            f"Target: {_single_line(_target_display(run_record))}",
            f"LLM Model: {_single_line(_llm_model(run_record))}",
            "App: strix",
            f"App Version: {_single_line(_app_version(run_record))}",
            f"Date (UTC): {now.strftime('%Y-%m-%d')}",
            "",
            "## Results",
            "",
            _README_RESULTS_HEADER,
            _README_RESULTS_ALIGNMENT,
            row,
            "",
        ]
    )


def write_input_artifacts(run_dir: Path, run_record: dict[str, Any]) -> dict[str, str]:
    """Copy configured user input files into ``run_dir``.

    The run record stores absolute source paths for reproducibility and
    relative artifact names for quick discovery inside the report directory.
    """
    existing = run_record.get("input_artifacts")
    copied: dict[str, str] = (
        {str(k): str(v) for k, v in existing.items()} if isinstance(existing, dict) else {}
    )

    for record_key, filename_prefix in _INPUT_ARTIFACTS:
        source_value = run_record.get(record_key)
        if not isinstance(source_value, str) or not source_value.strip():
            copied.pop(record_key, None)
            continue

        try:
            source_path = Path(source_value).expanduser().resolve(strict=True)
        except OSError as exc:
            logger.warning(
                "Could not copy %s artifact from %s: %s",
                filename_prefix,
                source_value,
                exc,
            )
            continue

        if not source_path.is_file():
            logger.warning(
                "Could not copy %s artifact from %s: not a file",
                filename_prefix,
                source_path,
            )
            continue

        target_path = run_dir / f"{filename_prefix}-{source_path.name}"
        try:
            if not _same_file(source_path, target_path):
                _atomic_copy2(source_path, target_path)
        except OSError as exc:
            logger.warning(
                "Could not copy %s artifact from %s to %s: %s",
                filename_prefix,
                source_path,
                target_path,
                exc,
            )
            continue

        copied[record_key] = target_path.name
        logger.info("Saved %s artifact to: %s", filename_prefix, target_path)

    if copied:
        run_record["input_artifacts"] = copied
    else:
        run_record.pop("input_artifacts", None)
    return copied


def write_vulnerabilities(
    run_dir: Path,
    vulnerability_reports: list[dict[str, Any]],
    saved_vuln_ids: set[str],
) -> int:
    vuln_dir = run_dir / "vulnerabilities"
    vuln_dir.mkdir(exist_ok=True)

    new_reports = [r for r in vulnerability_reports if r["id"] not in saved_vuln_ids]

    for report in new_reports:
        (vuln_dir / f"{report['id']}.md").write_text(
            render_vulnerability_md(report),
            encoding="utf-8",
        )
        saved_vuln_ids.add(report["id"])

    sorted_reports = sorted(
        vulnerability_reports,
        key=lambda r: (_SEVERITY_ORDER.get(r["severity"], 5), r["timestamp"]),
    )
    csv_path = run_dir / "vulnerabilities.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "title", "severity", "timestamp", "file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in sorted_reports:
            writer.writerow(
                {
                    "id": report["id"],
                    "title": report["title"],
                    "severity": report["severity"].upper(),
                    "timestamp": report["timestamp"],
                    "file": f"vulnerabilities/{report['id']}.md",
                },
            )

    _atomic_write_text(
        run_dir / "vulnerabilities.json",
        json.dumps(vulnerability_reports, ensure_ascii=False, indent=2, default=str),
    )

    if new_reports:
        logger.info(
            "Saved %d new vulnerability report(s) to: %s",
            len(new_reports),
            vuln_dir,
        )
    logger.info("Updated vulnerability index: %s", csv_path)
    return len(new_reports)


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    except OSError:
        _unlink_if_exists(tmp_path)
        raise


def _atomic_copy2(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(target_path.parent),
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        shutil.copy2(source_path, tmp_path)
        tmp_path.replace(target_path)
    except OSError:
        _unlink_if_exists(tmp_path)
        raise


def _unlink_if_exists(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not remove temporary artifact file: %s", path, exc_info=True)


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False


def _target_display(run_record: dict[str, Any]) -> str:
    targets = run_record.get("targets_info")
    if not isinstance(targets, list) or not targets:
        return "unknown"
    rendered = [_target_entry_display(target) for target in targets]
    return ", ".join(value for value in rendered if value) or "unknown"


def _target_entry_display(target: Any) -> str:
    if not isinstance(target, dict):
        return _single_line(target)
    original = target.get("original")
    if isinstance(original, str) and original.strip():
        return original
    details = target.get("details")
    if not isinstance(details, dict):
        return "unknown"
    for key in ("target_url", "target_repo", "target_path", "target_ip"):
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def _llm_model(run_record: dict[str, Any]) -> str:
    value = run_record.get("llm_model")
    if isinstance(value, str) and value.strip():
        return value
    return os.environ.get("STRIX_LLM", "") or "unknown"


def _app_version(run_record: dict[str, Any]) -> str:
    value = run_record.get("app_version")
    if isinstance(value, str) and value.strip():
        return value
    try:
        return package_version("strix-agent")
    except PackageNotFoundError:
        return "unknown"


def _elapsed_minutes(run_record: dict[str, Any], now: datetime) -> float:
    start = _parse_utc_datetime(run_record.get("start_time"))
    if start is None:
        return 0.0
    end = _parse_utc_datetime(run_record.get("end_time")) or now
    return max(0.0, (end - start).total_seconds() / 60)


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _total_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    total = _int_or_zero(usage.get("total_tokens"))
    if total:
        return total
    return _int_or_zero(usage.get("input_tokens")) + _int_or_zero(usage.get("output_tokens"))


def _count_tool_calls(run_dir: Path) -> int:
    agents_db = runtime_state_dir(run_dir) / "agents.db"
    if not agents_db.exists():
        return 0
    call_ids: set[str] = set()
    try:
        with sqlite3.connect(f"file:{agents_db}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "select id, message_data from agent_messages order by id"
            ).fetchall()
    except sqlite3.Error:
        logger.exception("Failed to count tool calls from %s", agents_db)
        return 0

    for row_id, message_data in rows:
        try:
            item = json.loads(message_data)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(item, dict) or item.get("type") not in {
            "function_call",
            "tool_call_item",
        }:
            continue
        call_id = item.get("call_id") or item.get("id") or row_id
        call_ids.add(str(call_id))
    return len(call_ids)


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table_cell(value: Any) -> str:
    return _single_line(value).replace("|", "\\|")


def _single_line(value: Any) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return " ".join(text.split()) or "unknown"


def _int_or_zero(value: Any) -> int:
    try:
        result = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return result if result > 0 else 0


def _float_or_zero(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if result > 0 else 0.0


def render_vulnerability_md(report: dict[str, Any]) -> str:  # noqa: PLR0912, PLR0915
    lines: list[str] = [
        f"# {report.get('title', 'Untitled Vulnerability')}\n",
        f"**ID:** {report.get('id', 'unknown')}",
        f"**Severity:** {report.get('severity', 'unknown').upper()}",
        f"**Found:** {report.get('timestamp', 'unknown')}",
    ]

    metadata: list[tuple[str, Any]] = [
        ("Target", report.get("target")),
        ("Endpoint", report.get("endpoint")),
        ("Method", report.get("method")),
        ("CVE", report.get("cve")),
        ("CWE", report.get("cwe")),
    ]
    cvss = report.get("cvss")
    if cvss is not None:
        metadata.append(("CVSS", cvss))
    for label, value in metadata:
        if value:
            lines.append(f"**{label}:** {value}")

    lines.append("")
    lines.append("## Description\n")
    lines.append(report.get("description") or "No description provided.")
    lines.append("")

    if report.get("impact"):
        lines.append("## Impact\n")
        lines.append(str(report["impact"]))
        lines.append("")

    if report.get("technical_analysis"):
        lines.append("## Technical Analysis\n")
        lines.append(str(report["technical_analysis"]))
        lines.append("")

    if report.get("poc_description") or report.get("poc_script_code"):
        lines.append("## Proof of Concept\n")
        if report.get("poc_description"):
            lines.append(str(report["poc_description"]))
            lines.append("")
        if report.get("poc_script_code"):
            lines.append("```")
            lines.append(str(report["poc_script_code"]))
            lines.append("```")
            lines.append("")

    if report.get("code_locations"):
        lines.append("## Code Analysis\n")
        for i, loc in enumerate(report["code_locations"]):
            file_ref = loc.get("file", "unknown")
            line_ref = ""
            if loc.get("start_line") is not None:
                if loc.get("end_line") and loc["end_line"] != loc["start_line"]:
                    line_ref = f" (lines {loc['start_line']}-{loc['end_line']})"
                else:
                    line_ref = f" (line {loc['start_line']})"
            lines.append(f"**Location {i + 1}:** `{file_ref}`{line_ref}")
            if loc.get("label"):
                lines.append(f"  {loc['label']}")
            if loc.get("snippet"):
                lines.append(f"  ```\n  {loc['snippet']}\n  ```")
            if loc.get("fix_before") or loc.get("fix_after"):
                lines.append("\n  **Suggested Fix:**")
                lines.append("```diff")
                if loc.get("fix_before"):
                    lines.extend(f"- {ln}" for ln in str(loc["fix_before"]).splitlines())
                if loc.get("fix_after"):
                    lines.extend(f"+ {ln}" for ln in str(loc["fix_after"]).splitlines())
                lines.append("```")
            lines.append("")

    if report.get("remediation_steps"):
        lines.append("## Remediation\n")
        lines.append(str(report["remediation_steps"]))
        lines.append("")

    return "\n".join(lines)
