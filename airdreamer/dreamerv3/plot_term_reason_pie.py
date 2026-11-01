import argparse
import json
import math
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a pie chart of episode termination reasons from scores.jsonl. "
            "Adjacent records are both relabeled to 'collide' when "
            "step_diff == worker_diff."
        )
    )
    parser.add_argument(
        "--input",
        default="/home/bdi/yck/logdir_airdreamer/multi_200m/20260308T221738_main_[-30,30]/scores.jsonl",
        help="Path to the scores.jsonl file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output image path. Defaults to <train_dir>/term_reason/"
            "<input_stem>_term_reason_pie.svg."
        ),
    )
    parser.add_argument(
        "--step-range",
        default=None,
        help=(
            "Exclusive step range in the form start-end. "
            "For example: 100000-200000 keeps start < step < end."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    return parser.parse_args()


def parse_step_range(spec):
    if not spec:
        return None
    if "-" not in spec:
        raise ValueError(
            f"Invalid --step-range {spec!r}. Expected format like 100000-200000."
        )
    start_text, end_text = spec.split("-", 1)
    start = float(start_text) if start_text.strip() else None
    end = float(end_text) if end_text.strip() else None
    if start is not None and end is not None and start >= end:
        raise ValueError(
            f"Invalid --step-range {spec!r}. Expected start < end."
        )
    return (start, end)


def in_step_range(step, step_range):
    if step_range is None:
        return True
    start, end = step_range
    if start is not None and step <= start:
        return False
    if end is not None and step >= end:
        return False
    return True


def load_records(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "step" not in item:
                raise ValueError(f"Missing 'step' at line {lineno}.")
            if "episode/worker" not in item:
                raise ValueError(f"Missing 'episode/worker' at line {lineno}.")
            records.append(
                {
                    "line": lineno,
                    "step": float(item["step"]),
                    "worker": float(item["episode/worker"]),
                    "reason": str(item.get("episode/term_reason", "<missing>")),
                }
            )
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def apply_collide_rule(records):
    adjusted = [dict(record, adjusted_reason=record["reason"]) for record in records]
    collide_lines = set()
    for prev, curr in zip(adjusted, adjusted[1:]):
        step_diff = curr["step"] - prev["step"]
        worker_diff = curr["worker"] - prev["worker"]
        if math.isclose(step_diff, worker_diff, rel_tol=0.0, abs_tol=1e-9):
            collide_lines.add(prev["line"])
            collide_lines.add(curr["line"])
    for record in adjusted:
        if record["line"] in collide_lines:
            record["adjusted_reason"] = "collide"
    return adjusted


def count_reasons(records, step_range):
    filtered = [record for record in records if in_step_range(record["step"], step_range)]
    if not filtered:
        raise ValueError("No episodes found in the requested step range.")
    counts = Counter(record["adjusted_reason"] for record in filtered)
    return filtered, counts


def build_title(custom_title, total, step_range):
    if custom_title:
        return custom_title
    if step_range is None:
        return f"Episode Term Reasons (all steps, n={total})"
    start, end = step_range
    lo = "-inf" if start is None else int(start)
    hi = "inf" if end is None else int(end)
    return f"Episode Term Reasons ({lo} < step < {hi}, n={total})"


def step_range_suffix(step_range):
    if step_range is None:
        return "all_steps"
    start, end = step_range
    lo = "neg_inf" if start is None else str(int(start))
    hi = "pos_inf" if end is None else str(int(end))
    return f"steps_{lo}_{hi}"


def default_output_path(input_path, step_range):
    input_path = Path(input_path)
    outdir = input_path.parent / "term_reason"
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = step_range_suffix(step_range)
    return outdir / f"{input_path.stem}_term_reason_pie_{suffix}.svg"


def plot_pie(counts, output, title):
    labels_and_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    values = [value for _, value in labels_and_counts]
    colors = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ab",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    total = sum(values)
    width = 1180
    height = 760
    cx = 300
    cy = 390
    radius = 220
    legend_x = 610
    legend_y = 150
    legend_step = 34

    def point(angle_deg, r):
        angle_rad = math.radians(angle_deg)
        return (
            cx + r * math.cos(angle_rad),
            cy + r * math.sin(angle_rad),
        )

    def sector_path(start_deg, end_deg):
        x1, y1 = point(start_deg, radius)
        x2, y2 = point(end_deg, radius)
        large_arc = 1 if (end_deg - start_deg) > 180.0 else 0
        return (
            f"M {cx:.3f} {cy:.3f} "
            f"L {x1:.3f} {y1:.3f} "
            f"A {radius:.3f} {radius:.3f} 0 {large_arc} 1 {x2:.3f} {y2:.3f} Z"
        )

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2:.1f}" y="48" text-anchor="middle" '
            f'font-size="26" font-family="sans-serif" font-weight="bold">'
            f"{escape(title)}</text>"
        ),
    ]

    if len(values) == 1:
        color = colors[0]
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}" '
            'stroke="white" stroke-width="2"/>'
        )
    else:
        current_angle = -90.0
        for index, (label, value) in enumerate(labels_and_counts):
            span = 360.0 * value / total
            start_angle = current_angle
            end_angle = current_angle + span
            color = colors[index % len(colors)]
            parts.append(
                f'<path d="{sector_path(start_angle, end_angle)}" fill="{color}" '
                'stroke="white" stroke-width="2"/>'
            )

            mid_angle = (start_angle + end_angle) / 2.0
            pct = 100.0 * value / total
            if pct >= 3.0:
                tx, ty = point(mid_angle, radius * 0.62)
                parts.append(
                    f'<text x="{tx:.3f}" y="{ty:.3f}" text-anchor="middle" '
                    'dominant-baseline="middle" font-size="14" '
                    'font-family="sans-serif" fill="black">'
                    f"{escape(f'{pct:.1f}% ({value})')}</text>"
                )
            current_angle = end_angle

    parts.append(
        (
            f'<text x="{legend_x}" y="{legend_y - 40}" font-size="20" '
            'font-family="sans-serif" font-weight="bold">Term Reason</text>'
        )
    )
    for index, (label, value) in enumerate(labels_and_counts):
        pct = 100.0 * value / total
        y = legend_y + index * legend_step
        color = colors[index % len(colors)]
        parts.append(
            f'<rect x="{legend_x}" y="{y - 16}" width="20" height="20" fill="{color}"/>'
        )
        parts.append(
            (
                f'<text x="{legend_x + 32}" y="{y}" font-size="18" '
                'font-family="sans-serif" dominant-baseline="middle">'
                f"{escape(f'{label}: {value} ({pct:.2f}%)')}</text>"
            )
        )

    parts.append("</svg>")
    Path(output).write_text("\n".join(parts), encoding="utf-8")


def print_summary(counts, total, step_range, output):
    if step_range is None:
        range_text = "all steps"
    else:
        start, end = step_range
        lo = "-inf" if start is None else int(start)
        hi = "inf" if end is None else int(end)
        range_text = f"{lo} < step < {hi}"
    print(f"Step range: {range_text}")
    print(f"Total episodes: {total}")
    print("Counts:")
    for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        pct = 100.0 * count / total
        print(f"  {label}: {count} ({pct:.2f}%)")
    print(f"Saved pie chart to: {output}")


def main():
    args = parse_args()
    step_range = parse_step_range(args.step_range)
    records = load_records(args.input)
    adjusted_records = apply_collide_rule(records)
    filtered_records, counts = count_reasons(adjusted_records, step_range)

    output = args.output
    if output is None:
        output = str(default_output_path(args.input, step_range))
    else:
        output = str(Path(output))
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    title = build_title(args.title, len(filtered_records), step_range)
    plot_pie(counts, output, title)
    print_summary(counts, len(filtered_records), step_range, output)


if __name__ == "__main__":
    main()
