import argparse
import json
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
        default="/home/bdi/yck/logdir_airdreamer/multi_200m/20260302T173406_main/scores.jsonl",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    df = read_jsonl(args.input)
    if "step" not in df.columns:
        raise ValueError("missing step column")
    if "episode/score" not in df.columns:
        raise ValueError("missing episode/score column")

    output = args.output
    if output is None:
        output = str(Path(args.input).with_suffix(".png"))

    df = df[["step", "episode/score"]].dropna()
    x = df["step"]
    y = df["episode/score"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, alpha=0.4, label="score")
    if args.smooth and args.smooth > 1:
        smooth = y.rolling(args.smooth, min_periods=1).mean()
        ax.plot(x, smooth, linewidth=2.0, label=f"score_ma{args.smooth}")

    ax.set_xlabel("step")
    ax.set_ylabel("episode/score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)


if __name__ == "__main__":
    main()
