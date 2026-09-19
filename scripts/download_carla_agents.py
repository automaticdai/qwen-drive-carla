"""Fetch only the official route planner and its imports, pinned to CARLA 0.9.16."""
from pathlib import Path
from urllib.request import urlopen

REVISION = "294096eb1c38eabf246e4f3a9cdab704e33a7f4c"
FILES = ("agents/navigation/global_route_planner.py", "agents/navigation/local_planner.py",
         "agents/navigation/controller.py", "agents/tools/misc.py")


def main():
    root = Path(__file__).resolve().parents[1] / "vendor/carla-agents"
    base = f"https://raw.githubusercontent.com/carla-simulator/carla/{REVISION}/"
    for relative, upstream in [(p, "PythonAPI/carla/" + p) for p in FILES] + [("LICENSE", "LICENSE")]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(base + upstream, timeout=30) as response:
            data = response.read()
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
    (root / "REVISION").write_text(REVISION + "\n")
    print(f"CARLA route helpers ready: {root}")


if __name__ == "__main__":
    main()
