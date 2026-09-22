"""End-to-end tests against the real HTTP server (stdlib urllib, no mocks)."""

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from app.server import Handler

# unittest discovery puts this directory on sys.path; the helpers are shared
# with the planner tests so the HTTP smoke checks the very same payload.
from test_planner import permute_request, reported_scenario


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

    def test_plan_reported_scenario_smoke(self):
        # The exact request from the bug report, over HTTP.
        status, body = self._post(reported_scenario())
        self.assertEqual(status, 200)
        self.assertEqual(body["objective"],
                         {"total_value": 16, "final_end_time": 7})
        self.assertEqual(body["canonical_plan"]["target_ids"], [10, 30, 20])
        self.assertEqual(body["optimal_target_set_count"], 1)
        self.assertFalse(body["empty_plan"])
        self.assertEqual(body["target_count"], 9)

        statuses = {c["id"]: c["status"] for c in body["classifications"]}
        self.assertEqual(
            statuses,
            {20: "required", 10: "required", 30: "required",
             40: "excluded", 41: "excluded", 42: "excluded",
             43: "excluded", 44: "excluded", 45: "excluded"},
        )

        # Recompute every step's slew/exposure/window evidence.
        steps = body["canonical_plan"]["steps"]
        self.assertEqual([s["order"] for s in steps], [1, 2, 3])
        self.assertEqual(
            [(s["id"], s["slew"]["from"], s["slew"]["seconds"],
              s["slew"]["finish_time"], s["start_time"], s["end_time"],
              s["exposure_seconds"],
              (s["window"]["open"], s["window"]["close"])) for s in steps],
            [(10, "NIGHT_START", 0, 0, 0, 1, 1, (0, 1)),
             (30, 10, 4, 5, 5, 6, 1, (0, 10)),
             (20, 30, 0, 6, 6, 7, 1, (0, 10))],
        )
        prev_end = 0
        prev_id = None
        raw = reported_scenario()
        by_id = {t["id"]: t for t in raw["targets"]}
        ids_in_order = [t["id"] for t in raw["targets"]]
        for s in steps:
            self.assertEqual(s["slew"]["finish_time"],
                             prev_end + s["slew"]["seconds"])
            # Slew seconds come from the request's own matrices.
            if prev_id is None:
                idx = ids_in_order.index(s["id"])
                want = raw["slew"]["from_night_start"][idx]
            else:
                i, j = ids_in_order.index(prev_id), ids_in_order.index(s["id"])
                want = raw["slew"]["between_targets"][i][j]
            self.assertEqual(s["slew"]["seconds"], want)
            self.assertGreaterEqual(s["start_time"], s["slew"]["finish_time"])
            self.assertEqual(s["end_time"],
                             s["start_time"] + s["exposure_seconds"])
            self.assertEqual(s["exposure_seconds"], by_id[s["id"]]["duration"])
            # The reported window is one of the target's request windows
            # and contains the whole exposure.
            self.assertIn(s["window"], by_id[s["id"]]["windows"])
            self.assertLessEqual(s["window"]["open"], s["start_time"])
            self.assertLessEqual(s["end_time"], s["window"]["close"])
            prev_end = s["end_time"]
            prev_id = s["id"]
        self.assertEqual(prev_end, body["objective"]["final_end_time"])

    def test_plan_reported_scenario_permuted_and_renumbered(self):
        reference_status, reference = self._post(reported_scenario())
        self.assertEqual(reference_status, 200)

        # Equivalent instance, targets array shuffled: identical business
        # result (canonical sequence, objective, classifications).
        raw = reported_scenario()
        perm = [4, 0, 8, 2, 6, 1, 3, 7, 5]
        status, body = self._post(permute_request(raw, perm))
        self.assertEqual(status, 200)
        self.assertEqual(body["objective"], reference["objective"])
        self.assertEqual(body["canonical_plan"], reference["canonical_plan"])
        self.assertEqual(body["optimal_target_set_count"],
                         reference["optimal_target_set_count"])
        self.assertEqual(
            {c["id"]: c["status"] for c in body["classifications"]},
            {c["id"]: c["status"] for c in reference["classifications"]},
        )

        # Same instance, ids renumbered: same business combination.
        id_map = {20: 7, 10: 900, 30: 55,
                  40: -3, 41: -2, 42: -1, 43: 0, 44: 1, 45: 2}
        renumbered = {
            "targets": [{**t, "id": id_map[t["id"]]} for t in raw["targets"]],
            "slew": raw["slew"],
        }
        status, body = self._post(renumbered)
        self.assertEqual(status, 200)
        self.assertEqual(body["objective"], reference["objective"])
        self.assertEqual(body["canonical_plan"]["target_ids"], [900, 55, 7])
        self.assertEqual(body["optimal_target_set_count"], 1)
        self.assertEqual(
            {c["id"]: c["status"] for c in body["classifications"]},
            {7: "required", 900: "required", 55: "required",
             -3: "excluded", -2: "excluded", -1: "excluded",
             0: "excluded", 1: "excluded", 2: "excluded"},
        )

    def test_unknown_route(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/nope")
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
