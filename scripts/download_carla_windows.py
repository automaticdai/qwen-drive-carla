"""Resume a parallel download of the official CARLA 0.9.16 Windows archive."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time

URL = "https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip"
CHUNK = 128 * 1024 * 1024


def main():
    root = Path(__file__).resolve().parents[1] / "downloads"
    root.mkdir(exist_ok=True)
    parts = root / "CARLA_0.9.16.parts"
    parts.mkdir(exist_ok=True)
    response = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--head", URL],
                              capture_output=True, text=True, check=True, timeout=30)
    headers = {key.lower(): value for key, value in
               (line.split(":", 1) for line in response.stdout.splitlines() if ":" in line)}
    size = int(headers["content-length"].strip())
    etag = headers["etag"].strip()
    identity = dict(url=URL, size=size, etag=etag, chunk_bytes=CHUNK)
    manifest = parts / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError("Archive changed; move the old parts directory before retrying")
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    archive = root / "CARLA_0.9.16.zip"
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")

    def fetch(index):
        start, end = index * CHUNK, min((index + 1) * CHUNK, size) - 1
        target = parts / f"{index:04d}.part"
        expected = end - start + 1
        if target.exists() and target.stat().st_size == expected:
            return expected
        for attempt in range(3):
            try:
                temporary = target.with_suffix(".tmp")
                header_file = target.with_suffix(".headers")
                subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                                "--range", f"{start}-{end}", "--header", f"If-Match: {etag}",
                                "--dump-header", str(header_file), "--output", str(temporary), URL],
                               check=True, timeout=180)
                header_text = header_file.read_text().lower()
                if f"content-range: bytes {start}-{end}/{size}" not in header_text:
                    raise RuntimeError("Server did not honor the requested byte range")
                if temporary.stat().st_size != expected:
                    raise RuntimeError("Incomplete download chunk")
                temporary.replace(target)
                return expected
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    count = (size + CHUNK - 1) // CHUNK
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for task in as_completed([pool.submit(fetch, i) for i in range(count)]):
            done += task.result()
            print(f"Downloaded {done / 1e9:.2f}/{size / 1e9:.2f} GB", flush=True)
    digest = hashlib.sha256()
    temporary = archive.with_suffix(".zip.tmp")
    with temporary.open("wb") as output:
        for i in range(count):
            with (parts / f"{i:04d}.part").open("rb") as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
    temporary.replace(archive)
    # This local digest detects later changes; it is not an upstream signature.
    (root / "CARLA_0.9.16.download.json").write_text(json.dumps(
        {**identity, "sha256": digest.hexdigest()}, indent=2) + "\n")
    print(f"Archive ready: {archive}", flush=True)


if __name__ == "__main__":
    main()
