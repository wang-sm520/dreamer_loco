import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_jsonl(path):
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        records = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/bdi/yck/logdir_airdreamer/test_nav/20260322T110834/metrics.jsonl",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument(
        "--avg-window",
        type=int,
        default= 50,
        help="每 n 条数据做一次分组平均；<=0 时自动计算",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=300,
        help="自动窗口时期望保留的点数上限",
    )
    return parser.parse_args()


def choose_avg_window(length, avg_window, target_points):
    if length <= 0:
        return 1
    if avg_window and avg_window > 0:
        return int(avg_window)
    if target_points <= 0:
        target_points = 300
    if length <= target_points:
        return 1
    return max(1, int((length + target_points - 1) // target_points))


def average_by_window(series, window):
    if window <= 1 or len(series) <= 1:
        return series
    grouped = series.reset_index(drop=True).copy()
    grouped["_group_id"] = grouped.index // int(window)
    return grouped.groupby("_group_id", as_index=False)[["x", "y"]].mean()


def main():
    args = parse_args()
    df = read_jsonl(args.input)
    keys = list(df.columns)
    if not keys:
        raise ValueError("no available keys to plot")

    if args.output:
        outdir = Path(args.output)
        if outdir.suffix:
            outdir = outdir.parent
        if not outdir.exists():
            outdir = Path(args.input).parent
    else:
        outdir = Path(args.input).parent

    step_values = None
    if "step" in df.columns:
        step_values = pd.to_numeric(df["step"], errors="coerce")

    for key in keys:
        safe_key = re.sub(r"[^a-zA-Z0-9._-]+", "_", key).strip("_")
        picture_dir = outdir / "picture"
        picture_dir.mkdir(parents=True, exist_ok=True)
        output = picture_dir / f"{safe_key}.png"

        if key == "step":
            y = pd.to_numeric(df[key], errors="coerce")
            x = pd.Series(range(len(y)))
            series = pd.DataFrame({"x": x, "y": y}).dropna()
            xlabel = "index"
        else:
            y = pd.to_numeric(df[key], errors="coerce")
            if step_values is not None:
                series = pd.DataFrame({"x": step_values, "y": y}).dropna()
                xlabel = "step"
            else:
                x = pd.Series(range(len(y)))
                series = pd.DataFrame({"x": x, "y": y}).dropna()
                xlabel = "index"

        fig, ax = plt.subplots(figsize=(10, 4))
        if len(series):
            avg_window = choose_avg_window(len(series), args.avg_window, args.target_points)
            series = average_by_window(series, avg_window)
            x = series["x"]
            y = series["y"]
            if args.smooth and args.smooth > 1:
                y = y.rolling(args.smooth, min_periods=1).mean()
            ax.plot(x, y)
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
