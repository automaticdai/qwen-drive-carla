"""Loopback-only live debug display and explicit run controls."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time


class Dashboard:
    def __init__(self, port=8877):
        self.condition = threading.Condition()
        self.state = dict(phase='starting', wall_updated=time.time())
        self.paused, self.stopped, self.restart, self.steps = False, False, False, 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, content, kind, status=200):
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == '/':
                    self.send(Path(__file__).with_name('dashboard.html').read_bytes(), 'text/html; charset=utf-8')
                elif self.path == '/state':
                    with owner.condition:
                        value = {**owner.state, 'paused': owner.paused, 'stop_requested': owner.stopped}
                    self.send(json.dumps(value, allow_nan=False).encode(), 'application/json')
                else:
                    self.send(b'Not found', 'text/plain', 404)

            def do_POST(self):
                # Reject cross-origin browser submissions and non-JSON form requests.
                origin = self.headers.get('Origin')
                if self.path != '/control' or self.headers.get('Content-Type') != 'application/json' or (origin and origin != 'http://' + self.headers.get('Host', '')):
                    self.send(b'Rejected', 'text/plain', 403); return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size < 128:
                        raise ValueError('Invalid size')
                    action = json.loads(self.rfile.read(size))['action']
                    owner.control(action)
                except (ValueError, KeyError, TypeError):
                    self.send(b'Invalid action', 'text/plain', 400); return
                self.send(b'{}', 'application/json')

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def publish(self, **values):
        with self.condition:
            self.state.update(values, wall_updated=time.time())

    def control(self, action):
        with self.condition:
            if action == 'pause':
                self.paused = True
            elif action == 'run':
                self.paused = False
            elif action == 'step':
                self.paused = True; self.steps += 1
            elif action == 'stop':
                self.stopped = True
            elif action == 'restart':
                self.restart = True; self.stopped = True
            else:
                raise ValueError('Unknown action')
            self.condition.notify_all()

    def allow_tick(self):
        with self.condition:
            while self.paused and self.steps == 0 and not self.stopped:
                self.condition.wait(timeout=.5)
            if self.stopped:
                return False
            if self.steps:
                self.steps -= 1
            return True

    def close(self):
        self.server.shutdown(); self.server.server_close()
        self.thread.join(timeout=5)
