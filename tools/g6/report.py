"""Render the validated G6 aggregate JSON as a human-reviewable Markdown report."""

from __future__ import annotations

from typing import Any


def _metric(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _mean_std(summary: dict[str, Any]) -> str:
    return f"{_metric(summary['mean'])} ± {_metric(summary['sample_std'])}"


def render_results_markdown(
    result: dict[str, Any], *, title: str = "G6 三数据集 Re-ID 正式结果"
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Protocol hash: `{result['protocol_hash']}`",
        f"- Git commit: `{result['git_commit']}`",
        f"- Validated evaluations: {int(result['num_evaluations'])}",
        "",
        "## Source-domain FrameAcc",
        "",
        "| Source | Seed 0 | Seed 42 | Seed 123 | Mean ± sample std |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in ("totalcapture", "egohumans"):
        summary = result["source"][source]
        values = summary["by_seed"]
        lines.append(
            f"| {source} | {_metric(values['0'])} | {_metric(values['42'])} | "
            f"{_metric(values['123'])} | {_mean_std(summary)} |"
        )

    lines.extend(
        [
            "",
            "## Custom 逐 session FrameAcc",
            "",
            "| Condition | Source | Test session | Seed 0 | Seed 42 | Seed 123 | Mean ± sample std | Counts by seed |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    condition_order = {"zero_shot": 0, "finetune": 1, "direct": 2}
    session_rows = sorted(
        result["custom_by_session"].values(),
        key=lambda row: (
            condition_order[str(row["condition"])],
            str(row.get("source") or "none"),
            str(row["session"]),
        ),
    )
    for row in session_rows:
        values = row["by_seed"]
        counts = row["counts_by_seed"]
        count_text = "; ".join(
            f"{seed}: {counts[seed]['correct']}/{counts[seed]['total']}"
            for seed in ("0", "42", "123")
        )
        lines.append(
            f"| {row['condition']} | {row.get('source') or 'none'} | {row['session']} | "
            f"{_metric(values['0'])} | {_metric(values['42'])} | {_metric(values['123'])} | "
            f"{_mean_std(row)} | {count_text} |"
        )

    lines.extend(
        [
            "",
            "## Custom 整体结果",
            "",
            "| Condition | Source | Macro-session | Micro/weighted | Session sample std |",
            "|---|---|---:|---:|---:|",
        ]
    )
    overall_rows = sorted(
        result["custom_overall"].values(),
        key=lambda row: (
            condition_order[str(row["condition"])],
            str(row.get("source") or "none"),
        ),
    )
    for row in overall_rows:
        lines.append(
            f"| {row['condition']} | {row.get('source') or 'none'} | "
            f"{_mean_std(row['macro_session'])} | {_mean_std(row['micro_weighted'])} | "
            f"{_mean_std(row['session_sample_std'])} |"
        )

    lines.extend(
        [
            "",
            "## 探索性配对比较",
            "",
            "差值方向为 left − right；bootstrap 以 12 个 seed×session 配对单元重采样。",
            "",
            "| Comparison | Mean difference | Sample std | Cohen dz | Bootstrap 95% CI | N |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, comparison in sorted(result["paired_comparisons"].items()):
        ci = comparison["bootstrap_mean_95ci"]
        effect = comparison["cohen_dz"]
        effect_text = "undefined" if effect is None else f"{float(effect):.3f}"
        lines.append(
            f"| {name} | {_metric(comparison['mean_paired_difference'])} | "
            f"{_metric(comparison['sample_std_difference'])} | {effect_text} | "
            f"[{_metric(ci['low'])}, {_metric(ci['high'])}] | {comparison['n_pairs']} |"
        )

    lines.extend(["", "所有数值均由机器可读 `correct/total` 与正式 run records 自动生成。", ""])
    return "\n".join(lines)
