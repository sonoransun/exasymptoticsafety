"""REST API task server for distributing computation work units.

A lightweight HTTP server that manages work unit lifecycle:
- POST /submit: Submit a new work unit
- GET /fetch: Fetch a pending work unit (for workers)
- POST /result/{id}: Submit result for a work unit
- GET /result/{id}: Check result status
- GET /status: Server status

Can be used with BOINC wrappers, cloud VMs, or direct HTTP clients.

Usage:
    python -m asymsafety.compute.distributed.task_server --port 8765
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import OrderedDict

from asymsafety.compute.distributed.work_unit import WorkUnit


class TaskStore:
    """Thread-safe storage for work units."""

    def __init__(self):
        self._units: OrderedDict[str, WorkUnit] = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, unit: WorkUnit) -> str:
        with self._lock:
            self._units[unit.id] = unit
            return unit.id

    def fetch_pending(self) -> WorkUnit | None:
        with self._lock:
            for unit in self._units.values():
                if unit.status == "pending":
                    unit.status = "running"
                    return unit
            return None

    def set_result(self, unit_id: str, result, error: str | None = None):
        with self._lock:
            if unit_id in self._units:
                self._units[unit_id].result = result
                self._units[unit_id].error = error
                self._units[unit_id].status = "failed" if error else "complete"

    def get_unit(self, unit_id: str) -> WorkUnit | None:
        with self._lock:
            return self._units.get(unit_id)

    def status(self) -> dict:
        with self._lock:
            counts = {"pending": 0, "running": 0, "complete": 0, "failed": 0}
            for unit in self._units.values():
                counts[unit.status] = counts.get(unit.status, 0) + 1
            return {"total": len(self._units), **counts}


_store = TaskStore()


class TaskHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the task server."""

    def do_POST(self):
        if self.path == "/submit":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            unit = WorkUnit.from_json(body.decode())
            unit_id = _store.submit(unit)
            self._json_response({"id": unit_id, "status": "submitted"})

        elif self.path.startswith("/result/"):
            unit_id = self.path.split("/result/")[1]
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            _store.set_result(unit_id, body.get("result"), body.get("error"))
            self._json_response({"status": "recorded"})

        else:
            self._json_response({"error": "not found"}, 404)

    def do_GET(self):
        if self.path == "/fetch":
            unit = _store.fetch_pending()
            if unit:
                self._json_response(json.loads(unit.to_json()))
            else:
                self._json_response({"status": "no_work"}, 204)

        elif self.path.startswith("/result/"):
            unit_id = self.path.split("/result/")[1]
            unit = _store.get_unit(unit_id)
            if unit:
                self._json_response({"status": unit.status, "result": unit.result, "error": unit.error})
            else:
                self._json_response({"error": "not found"}, 404)

        elif self.path == "/status":
            self._json_response(_store.status())

        else:
            self._json_response({"error": "not found"}, 404)

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


def run_server(port: int = 8765):
    """Start the task server."""
    server = HTTPServer(("0.0.0.0", port), TaskHandler)
    print(f"Task server running on http://0.0.0.0:{port}")
    print("Endpoints: POST /submit, GET /fetch, POST /result/<id>, GET /result/<id>, GET /status")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Asymptotic Safety task server")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.port)
