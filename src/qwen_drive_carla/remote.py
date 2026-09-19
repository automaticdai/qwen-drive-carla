"""Loopback-only Qwen inference transport, intended for an SSH local forward."""
import argparse
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
import uuid
import zlib

import numpy as np
from PIL import Image

from .adapter import COMMANDS, VIEWS, scene_payload
from .image_transport import ServerImageCache, ImageCacheMiss, unpack_body

VERSION = 1
MAX_REQUEST = 64 * 1024 * 1024
MAX_RESPONSE = 1024 * 1024
ARRAY_SHAPES = dict(history=(16, 3), history_velocity=(16, 2), history_acceleration=(16, 2),
                    ego_velocity=(2,), ego_acceleration=(2,))


def json_bytes(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':')).encode()


def parse_json(data):
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(data, parse_constant=invalid)


def encode_scene(payload, image_sizes=None):
    data = {key: np.asarray(payload[key]).tolist() for key in ARRAY_SHAPES}
    data.update(nav_command=payload['nav_command'], driving_command=payload['driving_command'], token=payload['token'])
    data['views'] = {}
    for tag, images in payload['views'].items():
        encoded = []
        for index, image in enumerate(images):
            if image_sizes and image.width > image_sizes[index][0]:
                # Upstream uses torchvision's PIL BICUBIC resize for target_size.
                # Resize history before upload, avoiding redundant full-resolution PNGs.
                image = image.resize(tuple(image_sizes[index]), Image.Resampling.BICUBIC)
            output = BytesIO()
            image.save(output, format='PNG')
            encoded.append(base64.b64encode(output.getvalue()).decode('ascii'))
        data['views'][tag] = encoded
    return data


def decode_scene(data):
    if not isinstance(data, dict):
        raise ValueError('Scene must be an object')
    payload = {}
    for key, shape in ARRAY_SHAPES.items():
        array = np.asarray(data[key], dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Invalid scene array: ' + key)
        payload[key] = array
    nav = data['nav_command']
    valid_commands = {n: command for n, command in COMMANDS.values()}
    if type(nav) is not int or nav not in valid_commands or data['driving_command'] != valid_commands[nav]:
        raise ValueError('Navigation command does not match driving command')
    if not isinstance(data['token'], str) or not 1 <= len(data['token']) <= 64:
        raise ValueError('Invalid frame token')
    payload.update(nav_command=nav, driving_command=valid_commands[nav], token=data['token'])
    if not isinstance(data['views'], dict) or set(data['views']) != set(VIEWS.values()):
        raise ValueError('Exactly three camera views are required')
    payload['views'] = {}
    for tag, images in data['views'].items():
        if not isinstance(images, list) or len(images) != 4:
            raise ValueError('Four history images per view are required')
        decoded = []
        for encoded in images:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(BytesIO(raw)) as image:
                if image.format not in ('PNG', 'WEBP', 'JPEG') or image.size not in ((320, 192), (640, 384), (1280, 768)):
                    raise ValueError('Images must be supported 5:3 PNG/WebP/JPEG sizes up to 1280x768')
                decoded.append(image.convert('RGB'))
        payload['views'][tag] = decoded
    return payload


class RemotePlanner:
    """No torch or CUDA dependency on the simulator/client machine."""
    def __init__(self, url, timeout=120):
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or
                parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/')):
            raise ValueError('Use an HTTP loopback URL through an SSH tunnel')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Timeout must be finite and positive')
        self.url, self.timeout = url.rstrip('/'), timeout
        self.opener = build_opener(ProxyHandler({}))
        health = self.request('/health')
        if health.get('protocol') != VERSION or health.get('status') != 'ready':
            raise RuntimeError('Remote planner is not ready or uses a different protocol')
        self.loading_info = {**health['model_profile'], 'transport': 'ssh-http-png', 'endpoint': self.url}

    def request(self, path, body=None, content_encoding=None):
        headers = {'Content-Type': 'application/json'}
        if content_encoding:
            headers['Content-Encoding'] = content_encoding
        request = Request(self.url + path, data=body, headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                data = response.read(MAX_RESPONSE + 1)
            if len(data) > MAX_RESPONSE:
                raise ValueError('Oversized planner response')
            return parse_json(data)
        except HTTPError as exc:
            detail = exc.read(4096).decode('utf-8', errors='replace')
            if exc.code == 409 and 'image_cache_miss' in detail:
                raise ImageCacheMiss(detail) from exc
            raise RuntimeError(f'Remote planner HTTP {exc.code}: {detail}') from exc
        except Exception as exc:
            raise RuntimeError(f'Remote planner request failed: {exc}') from exc

    def plan(self, records):
        started = time.perf_counter()
        payload = scene_payload(records, len(records) - 1)
        request_id = uuid.uuid4().hex
        body = json_bytes(dict(protocol=VERSION, request_id=request_id,
                               scene=encode_scene(payload, self.loading_info.get('image_sizes'))))
        if len(body) > MAX_REQUEST:
            raise ValueError('Scene exceeds transport size limit')
        response = self.request('/plan', body)
        if (response.get('protocol') != VERSION or response.get('request_id') != request_id or
                response.get('token') != payload['token']):
            raise RuntimeError('Remote response does not match the requested frame')
        trajectory = np.asarray(response['trajectory'], dtype=float)
        if trajectory.shape != (50, 3) or not np.isfinite(trajectory).all():
            raise ValueError('Remote trajectory must contain 50 finite XY/heading rows')
        metrics = response['metrics']
        return trajectory, {**metrics, 'inference_seconds': metrics['seconds'],
                            'seconds': time.perf_counter() - started, 'request_bytes': len(body),
                            'transport': 'ssh-http-png'}


def make_server(planner, port=8765):
    lock = threading.Lock()
    image_cache = ServerImageCache()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(120)

        def log_message(self, format, *args):
            pass  # Do not log camera payloads or per-request headers.

        def send_json(self, status, value):
            data = json_bytes(value)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != '/health':
                self.send_json(404, {'error': 'Not found'})
                return
            self.send_json(200, dict(protocol=VERSION, status='ready', model_profile=planner.loading_info))

        def do_POST(self):
            if self.path not in ('/plan', '/debug') or (self.path == '/debug' and not hasattr(planner, 'debug_payload')):
                self.send_json(404, {'error': 'Not found'})
                return
            if not lock.acquire(blocking=False):
                self.send_json(503, {'error': 'Planner busy; no request was queued'})
                return
            try:
                self.handle_plan()
            finally:
                lock.release()

        def handle_plan(self):
            request_started = time.perf_counter()
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_REQUEST:
                    self.send_json(413, {'error': 'Invalid request size'})
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('Incomplete request')
                request = parse_json(unpack_body(body, self.headers.get('Content-Encoding')))
                if request['protocol'] != VERSION or not isinstance(request['request_id'], str) or not 1 <= len(request['request_id']) <= 64:
                    raise ValueError('Invalid protocol or request ID')
                scene_data, surround_data = request['scene'], request.get('surround')
                cached_transport = self.path == '/debug' and 'image_uploads' in request
                if cached_transport:
                    scene_data, surround_data = image_cache.resolve(request)
                payload = decode_scene(scene_data)
                if self.path == '/debug':
                    from .debug_inference import decode_surround
                    surround = decode_surround(surround_data)
            except ImageCacheMiss:
                self.send_json(409, {'error': 'image_cache_miss; inference was not executed'})
                return
            except (ValueError, KeyError, TypeError, OSError, binascii.Error, zlib.error) as exc:
                self.send_json(400, {'error': str(exc)})
                return
            try:
                decode_seconds = time.perf_counter() - request_started
                extra = {}
                if self.path == '/debug':
                    trajectory, metrics, bev = planner.debug_payload(payload, surround)
                    extra['bev'] = bev
                    if cached_transport:
                        extra['cached_image_ids'] = list(image_cache.images)
                else:
                    trajectory, metrics = planner.plan_payload(payload)
                trajectory = np.asarray(trajectory, dtype=float)
                if trajectory.shape != (50, 3) or not np.isfinite(trajectory).all():
                    raise ValueError('Model returned a malformed or non-finite trajectory')
                metrics = {**metrics, 'server_decode_seconds': decode_seconds,
                           'server_request_seconds': time.perf_counter() - request_started}
                self.send_json(200, dict(protocol=VERSION, request_id=request['request_id'],
                                        token=payload['token'], trajectory=trajectory.tolist(), metrics=metrics, **extra))
            except Exception as exc:
                self.send_json(500, {'error': f'{type(exc).__name__}: {exc}'})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/Qwen-Drive-1.0-4B'))
    parser.add_argument('--precision', choices=('bf16', 'nf4'), default='nf4')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--planner', choices=('sft', 'rl'), default='sft')
    parser.add_argument('--planning-mode', choices=('direct', 'reasoning'), default='direct')
    parser.add_argument('--image-profile', choices=('small', 'high'), default='small')
    args = parser.parse_args()
    from .planner import QwenPlanner
    planner = QwenPlanner(args.model, precision=args.precision, planner=args.planner,
                          planning_mode=args.planning_mode, image_profile=args.image_profile)
    with make_server(planner, args.port) as server:
        print(f'Qwen ready on http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
