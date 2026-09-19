"""Download the pinned official VLM and SFT planner; safe to resume."""
from pathlib import Path
import argparse

from huggingface_hub import snapshot_download

REVISION = "28484089a7cc8c335cf5089fb0745cf7c49b6eaa"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-rl", action="store_true", help="Also download the reasoning-trained RL planning expert")
    parser.add_argument("--include-perception", action="store_true", help="Also download the BEV perception head")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[1] / "models/Qwen-Drive-1.0-4B"
    snapshot_download("Qwen/Qwen-Drive-1.0-4B", revision=REVISION,
                      local_dir=destination, ignore_patterns=([] if args.include_perception else ["perception/*"]) + ([] if args.include_rl else ["planner-rl/*"]))
    (destination / "REVISION").write_text(REVISION + "\n")
