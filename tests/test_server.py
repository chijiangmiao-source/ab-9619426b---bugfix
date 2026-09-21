"""End-to-end tests against the real HTTP server (stdlib urllib, no mocks)."""

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from app.server import Handler


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["API_PORT"] = str(_free_port())
        cls.port = int(os.environ["API_PORT"])
        cls.server = ThreadingHTTPServer(("127.0.0.1", cls.port), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _post(self, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/plan",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health(self):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/health"
        ) as resp:
            self.assertEqual(resp.status, 200)
            body = json.loads(resp.read())
            self.assertEqual(body["status"], "ok")

    def test_plan_ok(self):
        raw = {
            "targets": [
                {"id": 1, "duration": 3, "value": 5,
                 "windows": [{"open": 0, "close": 10}]},
                {"id": 2, "duration": 3, "value": 8,
                 "windows": [{"open": 0, "close": 10}]},
            ],
            "slew": {"from_night_start": [0, 0],
                     "between_targets": [[0, 1], [1, 0]]},
        }
        status, body = self._post(raw)
        self.assertEqual(status, 200)
        self.assertEqual(body["objective"]["total_value"], 13)
        self.assertEqual(body["canonical_plan"]["target_ids"], [1, 2])
        self.assertEqual(len(body["classifications"]), 2)

    def test_plan_invalid_json(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/plan",
            data=b"{not json",
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)
        self.assertEqual(json.loads(cm.exception.read())["error"], "invalid_json")

    def test_plan_invalid_payload_rejected_wholesale(self):
        raw = {
            "targets": [
                {"id": 1, "duration": 0, "value": 5,
                 "windows": [{"open": 0, "close": 10}]},
                {"id": 2, "duration": 3, "value": 8,
                 "windows": [{"open": 0, "close": 10}]},
            ],
            "slew": {"from_night_start": [0, 0],
                     "between_targets": [[0, 1], [1, 0]]},
        }
        status, body = self._post(raw)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_request")
        self.assertIn("duration", body["detail"])

    def test_unknown_route(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/nope")
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
