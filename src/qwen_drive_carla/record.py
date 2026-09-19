"""Capture synchronized cameras and ego motion from a dedicated CARLA server."""
import argparse
import json
import math
from pathlib import Path

from .adapter import COMMANDS
from .session import CarlaSession, image_for_frame, save_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", choices=COMMANDS, default="straight",
                        help="Fixed model intent, not inferred from autopilot. Choose a matching road segment.")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not math.isfinite(args.timeout) or args.seconds < 1.6 or args.timeout <= 0:
        parser.error("Use at least 1.6 seconds and a positive timeout")
    args.output.mkdir(parents=True, exist_ok=False)
    session = CarlaSession(args.host, args.port, args.tm_port, args.spawn_index, args.seed, args.timeout)
    metadata = {"command": args.command, "command_source": "user_fixed", "complete": False}
    try:
        with session:
            session.autopilot(True)
            with (args.output / "frames.jsonl").open("w") as handle:
                for _ in range(round(args.seconds * 10)):
                    save_record(session.tick(args.command), args.output, handle)
            metadata["complete"] = True
        print(f"Saved recording to {args.output}")
    finally:
        (args.output / "metadata.json").write_text(json.dumps({**session.metadata, **metadata}, indent=2) + "\n")


if __name__ == "__main__":
    main()
