"""Plot route progress and speed, including handover and raw lane events."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.episode / "steps.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("Episode has no recorded steps")
    summary = json.loads((args.episode / "summary.json").read_text())
    t = np.array([r["timestamp"] for r in rows]) - rows[0]["timestamp"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    axes[0].plot(t, [r["speed_mps"] for r in rows], label="Measured speed")
    target = [r["control"]["target_speed"] if r["driver"] == "model" else np.nan for r in rows]
    axes[0].plot(t, target, label="Controller target", alpha=.8)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend()
    axes[1].plot(t, [100 * r["route"]["progress"] for r in rows], color="tab:green")
    axes[1].set(xlabel="Simulation time since first observation (s)", ylabel="Route progress (%)", ylim=(0, 100))
    handover = next((t[i] for i, r in enumerate(rows) if r["frame"] == summary.get("handover_frame")), None)
    lane_times = [t[i] for i, r in enumerate(rows) for e in r["events"] if e["kind"] == "lane_invasion"]
    for ax in axes:
        if handover is not None:
            ax.axvspan(0, handover, color="gray", alpha=.15)
            ax.axvline(handover, color="black", linestyle="--", linewidth=1)
        for timestamp in lane_times:
            ax.axvline(timestamp, color="tab:red", alpha=.6, linewidth=1)
        ax.grid(alpha=.25)
    fig.suptitle(f"{summary['status']} | {summary.get('model_distance_m', 0):.1f} m under model control\n"
                 f"Gray: autopilot warmup · red: lane-marking events ({summary['lane_invasion_events']})")
    output = args.episode / "episode.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
