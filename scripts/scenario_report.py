"""Build a local replay gallery from completed scenario-suite results."""
import argparse
import html
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
results = json.loads((args.directory / 'results.json').read_text())
parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><title>Qwen / CARLA scenario results</title>
<style>body{font:16px system-ui;background:#101820;color:#edf3fa;max-width:1200px;margin:32px auto;padding:0 20px}h1{margin-bottom:8px}p{color:#b7c8d8}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:24px}article{background:#1b2b39;padding:20px;border-radius:12px}video{width:100%;border-radius:8px}dl{display:grid;grid-template-columns:1fr 1fr;gap:8px}dd{margin:0}a{color:#83cfff}</style>
<h1>Qwen / CARLA junction tests</h1><p>Autopilot references and Qwen closed-loop runs. Replay uses simulation time; inference pauses are removed. Qwen starts after 1.5 seconds of autopilot warmup. Lane events and route completion are experimental metrics, not a traffic-rule certification.</p><main>''']
for result in results:
    name = result['name']
    parts.append('<article><h2>' + html.escape(name) + '</h2>')
    if result.get('review_note'):
        parts.append('<p><strong>Review: ' + html.escape(result['review_note']) + '</strong></p>')
    video = args.directory / name / 'front-replay.mp4'
    if video.exists():
        parts.append('<video controls preload="metadata" src="' + html.escape(name) + '/front-replay.mp4"></video>')
    parts.append('<dl>')
    values = [('Outcome', result['status']), ('Route progress', f"{result.get('progress', 0):.0%}"),
              ('Model distance', f"{result.get('model_distance_m', 0):.1f} m"),
              ('Collisions', result.get('collision_events', '—')), ('Lane events', result.get('lane_invasion_events', '—')),
              ('Red-light crossings', result.get('red_light_events', 'Not scored')),
              ('Ambiguous signal crossings', result.get('signal_ambiguous_events', 'Not scored')),
              ('Signal check available', str(result.get('signal_check_available', False))),
              ('Clean completion', str(result.get('clean_completion', False))),
              ('Process exit code', result.get('exit_code', 'Unknown')),
              ('Cleanup verified', str(result.get('cleanup_verified', False))),
              ('Vehicles spawned', result.get('scene', {}).get('traffic_spawned', 'Not recorded')),
              ('Closest traffic centre', f"{result['nearest_traffic_center_m']:.1f} m" if result.get('nearest_traffic_center_m') is not None else 'None')]
    for key, value in values:
        parts.append('<dt>' + html.escape(key) + '</dt><dd>' + html.escape(str(value)) + '</dd>')
    parts.append('</dl></article>')
parts.append('</main></html>')
path = args.directory / 'report.html'
path.write_text('\n'.join(parts))
print(path)
