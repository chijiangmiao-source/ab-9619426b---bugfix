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

    @staticmethod
    def _nine_target_body():
        return {
            "targets": [
                {"id": 20, "duration": 1, "value": 10,
                 "windows": [{"open": 0, "close": 10}]},
                {"id": 10, "duration": 1, "value": 5,
                 "windows": [{"open": 0, "close": 1}]},
                {"id": 30, "duration": 1, "value": 1,
                 "windows": [{"open": 0, "close": 10}]},
                {"id": 40, "duration": 1, "value": 1,
                 "windows": [{"open": 0, "close": 1}]},
                {"id": 41, "duration": 1, "value": 1,
                 "windows": [{"open": 0, "close": 1}]},
                {"id": 42, "duration": 1, "value": 1,
                 "windows": [{"open": 0, "close": 1}]},
                {"id": 43, "duration": 1, "value": 1,
                 "windows": [{"open": 0, "close": 1}]},
                {"id": 44, "duration": 1, "value": 1,
                 "windows": [{"open": 0, "close": 1}]},
                {"id": 45, "duration": 1, "value": 1,
                 "windows": [{"open": 0, "close": 1}]},
            ],
            "slew": {
                "from_night_start": [0, 0, 100, 100, 100, 100, 100, 100, 100],
                "between_targets": [
                    [0, 100, 0, 100, 100, 100, 100, 100, 100],
                    [100, 0, 4, 100, 100, 100, 100, 100, 100],
                    [0, 100, 0, 100, 100, 100, 100, 100, 100],
                    [100, 100, 100, 0, 100, 100, 100, 100, 100],
                    [100, 100, 100, 100, 0, 100, 100, 100, 100],
                    [100, 100, 100, 100, 100, 0, 100, 100, 100],
                    [100, 100, 100, 100, 100, 100, 0, 100, 100],
                    [100, 100, 100, 100, 100, 100, 100, 0, 100],
                    [100, 100, 100, 100, 100, 100, 100, 100, 0],
                ],
            },
        }

    def test_nine_target_optimal_combo_smoke(self):
        # Regression submitted over real HTTP: the legal continuation
        # 10 -> 30 -> 20 (value 16) must beat the locally attractive 20,30.
        raw = self._nine_target_body()
        status, body = self._post(raw)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["objective"], {"total_value": 16, "final_end_time": 7}
        )
        self.assertEqual(body["optimal_target_set_count"], 1)
        self.assertFalse(body["empty_plan"])
        self.assertEqual(body["target_count"], 9)
        self.assertEqual(body["canonical_plan"]["target_ids"], [10, 30, 20])

        statuses = {c["id"]: c["status"] for c in body["classifications"]}
        self.assertEqual(len(body["classifications"]), 9)
        for tid in (10, 20, 30):
            self.assertEqual(statuses[tid], "required")
        for tid in range(40, 46):
            self.assertEqual(statuses[tid], "excluded")

        steps = body["canonical_plan"]["steps"]
        expected = [
            (1, 10, "NIGHT_START", 0, 0, 0, 1, 0, 1),
            (2, 30, 10, 4, 5, 5, 6, 0, 10),
            (3, 20, 30, 0, 6, 6, 7, 0, 10),
        ]
        prev_end = 0
        for step, (order, tid, frm, sec, fin, start, end, wo, wc) in zip(
            steps, expected
        ):
            self.assertEqual(step["order"], order)
            self.assertEqual(step["id"], tid)
            self.assertEqual(step["slew"]["from"], frm)
            self.assertEqual(step["slew"]["seconds"], sec)
            self.assertEqual(step["slew"]["finish_time"], fin)
            self.assertEqual(step["start_time"], start)
            self.assertEqual(step["end_time"], end)
            self.assertEqual(step["exposure_seconds"], end - start)
            self.assertEqual(step["window"], {"open": wo, "close": wc})
            # Re-checkable from the documented invariants alone.
            self.assertEqual(fin, prev_end + sec)
            self.assertGreaterEqual(start, fin)
            self.assertGreaterEqual(start, wo)
            self.assertLessEqual(end, wc)
            prev_end = end

    def test_nine_target_case_invariant_under_reordering(self):
        # Equivalent request with the targets array shuffled and slew rows
        # permuted consistently: identical business answer over HTTP.
        order = [6, 0, 8, 2, 4, 1, 7, 3, 5]  # 42,20,45,30,41,10,44,40,43
        base = self._nine_target_body()
        shuffled = {
            "targets": [base["targets"][i] for i in order],
            "slew": {
                "from_night_start": [
                    base["slew"]["from_night_start"][i] for i in order
                ],
                "between_targets": [
                    [base["slew"]["between_targets"][i][j] for j in order]
                    for i in order
                ],
            },
        }
        status, body = self._post(shuffled)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["objective"], {"total_value": 16, "final_end_time": 7}
        )
        self.assertEqual(body["canonical_plan"]["target_ids"], [10, 30, 20])
        self.assertEqual(body["optimal_target_set_count"], 1)
        statuses = {c["id"]: c["status"] for c in body["classifications"]}
        self.assertEqual({statuses[t] for t in (10, 20, 30)}, {"required"})
        self.assertTrue(
            all(statuses[t] == "excluded" for t in range(40, 46))
        )
        steps = body["canonical_plan"]["steps"]
        self.assertEqual(
            [(s["id"], s["slew"]["seconds"], s["start_time"], s["end_time"])
             for s in steps],
            [(10, 0, 0, 1), (30, 4, 5, 6), (20, 0, 6, 7)],
        )

    def test_nine_target_case_invariant_under_relabeling(self):
        # Same business request with renumbered ids: the same business
        # combination wins; only the id values change.
        remap = {20: 700, 10: 2, 30: 33, 40: 401, 41: 402, 42: 403,
                 43: 404, 44: 405, 45: 406}
        base = self._nine_target_body()
        relabeled = {
            "targets": [{**t, "id": remap[t["id"]]} for t in base["targets"]],
            "slew": base["slew"],
        }
        status, body = self._post(relabeled)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["objective"], {"total_value": 16, "final_end_time": 7}
        )
        self.assertEqual(body["optimal_target_set_count"], 1)
        # Lex rule 3 orders by id: 2 (was 10) -> 33 (was 30) -> 700 (was 20).
        self.assertEqual(body["canonical_plan"]["target_ids"], [2, 33, 700])
        steps = body["canonical_plan"]["steps"]
        self.assertEqual(
            [(s["id"], s["slew"]["from"], s["slew"]["seconds"],
              s["slew"]["finish_time"], s["start_time"], s["end_time"])
             for s in steps],
            [(2, "NIGHT_START", 0, 0, 0, 1),
             (33, 2, 4, 5, 5, 6),
             (700, 33, 0, 6, 6, 7)],
        )
        statuses = {c["id"]: c["status"] for c in body["classifications"]}
        for old, new in remap.items():
            want = "required" if old in (10, 20, 30) else "excluded"
            self.assertEqual(statuses[new], want)

    def test_unknown_route(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/nope")
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
