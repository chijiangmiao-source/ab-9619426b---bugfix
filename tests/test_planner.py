"""Tests for the planner, including a brute-force cross-check on small cases."""

import itertools
import random
import unittest

from app.planner import PlanError, Window, plan


def reported_scenario():
    """The 9-target request from the bug report: the optimum chains
    10 -> 30 -> 20 (value 16, end 7), which only exists via a follow-up
    combination that used to be pruned away on larger target sets."""
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


def permute_request(raw, perm):
    """Same instance, targets array reordered: perm[new_index] = old_index."""
    targets = raw["targets"]
    s0 = raw["slew"]["from_night_start"]
    sm = raw["slew"]["between_targets"]
    return {
        "targets": [targets[i] for i in perm],
        "slew": {
            "from_night_start": [s0[i] for i in perm],
            "between_targets": [[sm[i][j] for j in perm] for i in perm],
        },
    }


def earliest(windows, ready, duration):
    best = None
    for lo, hi in windows:
        t = max(ready, lo)
        if t + duration <= hi and (best is None or t < best):
            best = t
    return best


def brute(raw):
    """Reference enumeration; mirrors the documented rules exactly."""
    targets = raw["targets"]
    n = len(targets)
    s0 = raw["slew"]["from_night_start"]
    sm = raw["slew"]["between_targets"]
    d = [t["duration"] for t in targets]
    v = [t["value"] for t in targets]
    w = [[(x["open"], x["close"]) for x in t.get("windows", [])] for t in targets]
    ids = [t["id"] for t in targets]

    best_value, best_end = 0, 0
    best_seq = []
    optimal_sets = {frozenset()}
    candidates = [()]  # feasible permutations found so far; seed empty
    for r in range(1, n + 1):
        for perm in itertools.permutations(range(n), r):
            ends, ok = [], True
            prev_end, last = 0, None
            for pos, k in enumerate(perm):
                ready = s0[k] if pos == 0 else prev_end + sm[last][k]
                st = earliest(w[k], ready, d[k])
                if st is None:
                    ok = False
                    break
                prev_end = st + d[k]
                last = k
            if not ok:
                continue
            val = sum(v[k] for k in perm)
            end = prev_end
            seq = [ids[k] for k in perm]
            if (
                val > best_value
                or (val == best_value and end < best_end)
                or (val == best_value and end == best_end and seq < best_seq)
            ):
                best_value, best_end, best_seq = val, end, seq
            if val >= best_value:  # collect after global scan below
                pass
    # Second pass: collect every plan tied on value+end (sets only).
    opt_sets = set()
    if best_value == 0 and best_end == 0:
        opt_sets.add(frozenset())
    for r in range(1, n + 1):
        for perm in itertools.permutations(range(n), r):
            prev_end, last, ok = 0, None, True
            for pos, k in enumerate(perm):
                ready = s0[k] if pos == 0 else prev_end + sm[last][k]
                st = earliest(w[k], ready, d[k])
                if st is None:
                    ok = False
                    break
                prev_end = st + d[k]
                last = k
            if ok:
                val = sum(v[k] for k in perm)
                if val == best_value and prev_end == best_end:
                    opt_sets.add(frozenset(perm))
    return best_value, best_end, best_seq, opt_sets


def make_raw(n, d, v, w, s0, sm, ids=None):
    ids = ids or list(range(1, n + 1))
    return {
        "targets": [
            {
                "id": ids[i],
                "duration": d[i],
                "value": v[i],
                "windows": [{"open": a, "close": b} for a, b in w[i]],
            }
            for i in range(n)
        ],
        "slew": {"from_night_start": s0, "between_targets": sm},
    }


class TestPlanner(unittest.TestCase):
    def assertMatchesBrute(self, raw):
        res = plan(raw)
        bv, be, bseq, osets = brute(raw)
        self.assertEqual(res["objective"]["total_value"], bv)
        self.assertEqual(res["objective"]["final_end_time"], be)
        self.assertEqual(res["canonical_plan"]["target_ids"], bseq)
        self.assertEqual(res["optimal_target_set_count"], len(osets))
        ids = [t["id"] for t in raw["targets"]]
        id_to_sets = {tid: set() for tid in ids}
        for s in osets:
            for k in s:
                id_to_sets[ids[k]].add(frozenset(s))
        for entry in res["classifications"]:
            present = any(entry["id"] == ids[k] for s in osets for k in s)
            absent = any(entry["id"] != ids[k] for s in osets for k in s) or any(
                entry["id"] not in {ids[k] for k in s} for s in osets
            )
            want = (
                "required"
                if present and not any(
                    entry["id"] not in {ids[k] for k in s} for s in osets
                )
                else "optional"
                if present
                else "excluded"
            )
            self.assertEqual(entry["status"], want, entry)

    def test_combo_beats_greedy_priority(self):
        # Target 1: huge value but its block forecloses 2 and 3;
        # targets 2+3 together are worth more. Priority-greedy would fail.
        raw = make_raw(
            n=3,
            d=[10, 5, 5],
            v=[100, 60, 60],
            w=[[(0, 12)], [(0, 6)], [(6, 12)]],
            s0=[0, 0, 0],
            sm=[[0, 50, 50], [50, 0, 1], [50, 1, 0]],
        )
        res = plan(raw)
        # Target 1 alone: start 0 end 10 value 100. 2->3: 0..5 then slew 1,
        # ready 6, start 6 end 11, value 120.
        self.assertEqual(res["objective"]["total_value"], 120)
        self.assertEqual(res["objective"]["final_end_time"], 11)
        self.assertEqual(res["canonical_plan"]["target_ids"], [2, 3])
        self.assertMatchesBrute(raw)

    def test_value_then_end_then_lex(self):
        # Two mutually exclusive equal-value alternatives (cross slew too
        # long to chain); earlier end wins.
        raw = make_raw(
            n=2,
            d=[5, 5],
            v=[7, 7],
            w=[[(10, 20)], [(0, 10)]],
            s0=[0, 0],
            sm=[[0, 100], [100, 0]],
        )
        res = plan(raw)
        self.assertEqual((res["objective"]["total_value"],
                          res["objective"]["final_end_time"]), (7, 5))
        self.assertEqual(res["canonical_plan"]["target_ids"], [2])
        self.assertMatchesBrute(raw)

        # Equal value and end: smaller id sequence wins.
        raw = make_raw(
            n=2,
            d=[5, 5],
            v=[7, 7],
            w=[[(0, 10)], [(0, 10)]],
            s0=[0, 0],
            sm=[[0, 0], [0, 0]],
        )
        res = plan(raw)
        # Both fit: value 14 beats 7; lex smaller sequence is [1, 2].
        self.assertEqual(res["objective"]["total_value"], 14)
        self.assertEqual(res["canonical_plan"]["target_ids"], [1, 2])
        self.assertMatchesBrute(raw)

        # True lex tie: alternatives {3} and {1,2} share value and end.
        raw = make_raw(
            n=3,
            d=[2, 2, 4],
            v=[3, 4, 7],
            w=[[(0, 10)], [(0, 10)], [(0, 4)]],
            s0=[0, 0, 0],
            sm=[[0, 0, 100], [0, 0, 100], [100, 100, 0]],
        )
        res = plan(raw)
        self.assertEqual(res["objective"]["total_value"], 7)
        self.assertEqual(res["objective"]["final_end_time"], 4)
        self.assertEqual(res["canonical_plan"]["target_ids"], [1, 2])
        statuses = {c["id"]: c["status"] for c in res["classifications"]}
        self.assertEqual(statuses[1], "optional")
        self.assertEqual(statuses[2], "optional")
        self.assertEqual(statuses[3], "optional")
        self.assertMatchesBrute(raw)

    def test_required_target(self):
        # Target 1 is in every optimal plan; exactly one of 2 or 3 can join
        # (their windows close at t=2 and the exposure is 2s).
        raw = make_raw(
            n=3,
            d=[2, 2, 2],
            v=[100, 1, 1],
            w=[[(0, 10)], [(0, 2)], [(0, 2)]],
            s0=[0, 0, 0],
            sm=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        )
        res = plan(raw)
        statuses = {c["id"]: c["status"] for c in res["classifications"]}
        self.assertEqual(statuses[1], "required")
        self.assertEqual(statuses[2], "optional")
        self.assertEqual(statuses[3], "optional")
        # Feasible optima: [2,1] and [3,1]; lex smallest is [2,1].
        self.assertEqual(res["canonical_plan"]["target_ids"], [2, 1])
        self.assertMatchesBrute(raw)

    def test_via_other_target_faster_than_direct(self):
        # Direct slew to 2 is huge; via target 1 it is fast -- no triangle
        # inequality assumed.
        raw = make_raw(
            n=2,
            d=[1, 5],
            v=[1, 10],
            w=[[(0, 5)], [(0, 20)]],
            s0=[0, 100],
            sm=[[0, 1], [100, 0]],
        )
        res = plan(raw)
        self.assertEqual(res["objective"]["total_value"], 11)
        steps = res["canonical_plan"]["steps"]
        self.assertEqual([s["id"] for s in steps], [1, 2])
        self.assertEqual(steps[1]["slew"]["seconds"], 1)
        self.assertEqual(steps[1]["start_time"], 2)
        self.assertEqual(steps[1]["end_time"], 7)
        self.assertMatchesBrute(raw)

    def test_closed_integer_window_boundaries(self):
        # Exposure of 5 in [0, 5] must start exactly at 0 (end == close ok).
        raw = make_raw(
            n=2,
            d=[5, 5],
            v=[1, 1],
            w=[[(0, 5)], [(1, 5)]],  # second infeasible (1+5 > 5)
            s0=[0, 0],
            sm=[[0, 0], [0, 0]],
        )
        res = plan(raw)
        self.assertEqual(res["canonical_plan"]["steps"][0]["start_time"], 0)
        self.assertEqual(res["canonical_plan"]["steps"][0]["end_time"], 5)
        statuses = {c["id"]: c["status"] for c in res["classifications"]}
        self.assertEqual(statuses[2], "excluded")
        self.assertMatchesBrute(raw)

    def test_no_visible_targets_zero_conclusion(self):
        raw = make_raw(
            n=2,
            d=[5, 5],
            v=[1, 1],
            w=[[], []],
            s0=[0, 0],
            sm=[[0, 0], [0, 0]],
        )
        res = plan(raw)
        self.assertEqual(res["objective"], {"total_value": 0, "final_end_time": 0})
        self.assertTrue(res["empty_plan"])
        self.assertEqual(res["canonical_plan"]["target_ids"], [])
        self.assertEqual(res["canonical_plan"]["steps"], [])
        self.assertTrue(all(c["status"] == "excluded" for c in res["classifications"]))

    def test_windows_exist_but_slew_never_in_time(self):
        # Window closes at 3, exposure is 2, slew from night (and via the
        # other target) takes 10 -> nothing observable, zero conclusion.
        raw = make_raw(
            n=2,
            d=[2, 2],
            v=[5, 5],
            w=[[(0, 3)], [(0, 3)]],
            s0=[10, 10],
            sm=[[0, 10], [10, 0]],
        )
        res = plan(raw)
        self.assertEqual(res["objective"], {"total_value": 0, "final_end_time": 0})
        self.assertTrue(res["empty_plan"])
        self.assertTrue(all(c["status"] == "excluded" for c in res["classifications"]))
        self.assertEqual(res["canonical_plan"]["steps"], [])

    def test_negative_ids_and_nonmonotonic_numbering(self):
        raw = make_raw(
            n=3,
            d=[1, 1, 1],
            v=[1, 1, 1],
            w=[[(0, 5)]] * 3,
            s0=[0, 0, 0],
            sm=[[0] * 3 for _ in range(3)],
            ids=[10, -5, 0],
        )
        res = plan(raw)
        self.assertEqual(res["canonical_plan"]["target_ids"], [-5, 0, 10])
        self.assertMatchesBrute(raw)

    def test_step_evidence_fields(self):
        raw = make_raw(
            n=2,
            d=[4, 3],
            v=[5, 6],
            w=[[(2, 10)], [(0, 20)]],
            s0=[1, 2],
            sm=[[0, 2], [3, 0]],
        )
        res = plan(raw)
        steps = res["canonical_plan"]["steps"]
        s1, s2 = steps
        self.assertEqual(s1["slew"]["from"], "NIGHT_START")
        self.assertEqual(s1["slew"]["seconds"], 1)
        self.assertEqual(s1["slew"]["finish_time"], 1)
        self.assertEqual(s1["start_time"], 2)   # window opens at 2
        self.assertEqual(s1["end_time"], 6)
        self.assertEqual(s1["window"], {"open": 2, "close": 10})
        self.assertEqual(s2["slew"]["from"], 1)
        self.assertEqual(s2["slew"]["seconds"], 2)
        self.assertEqual(s2["slew"]["finish_time"], 8)
        self.assertEqual((s2["start_time"], s2["end_time"]), (8, 11))

    def test_reported_nine_target_scenario(self):
        # Regression: legal follow-up combinations were pruned on larger
        # target sets, losing the global optimum [10, 30, 20].
        res = plan(reported_scenario())
        self.assertEqual(res["objective"],
                         {"total_value": 16, "final_end_time": 7})
        self.assertEqual(res["canonical_plan"]["target_ids"], [10, 30, 20])
        self.assertEqual(res["optimal_target_set_count"], 1)
        self.assertFalse(res["empty_plan"])

        statuses = {c["id"]: c["status"] for c in res["classifications"]}
        self.assertEqual(
            statuses,
            {20: "required", 10: "required", 30: "required",
             40: "excluded", 41: "excluded", 42: "excluded",
             43: "excluded", 44: "excluded", 45: "excluded"},
        )

        # Every step's slew source, finish time, exposure interval and
        # window evidence recomputes from the request.
        steps = res["canonical_plan"]["steps"]
        self.assertEqual(
            [(s["id"], s["slew"]["from"], s["slew"]["seconds"],
              s["slew"]["finish_time"], s["start_time"], s["end_time"],
              (s["window"]["open"], s["window"]["close"])) for s in steps],
            [(10, "NIGHT_START", 0, 0, 0, 1, (0, 1)),
             (30, 10, 4, 5, 5, 6, (0, 10)),
             (20, 30, 0, 6, 6, 7, (0, 10))],
        )
        prev_end = 0
        for s in steps:
            self.assertEqual(s["slew"]["finish_time"],
                             prev_end + s["slew"]["seconds"])
            self.assertGreaterEqual(s["start_time"], s["slew"]["finish_time"])
            self.assertEqual(s["end_time"],
                             s["start_time"] + s["exposure_seconds"])
            self.assertLessEqual(s["window"]["open"], s["start_time"])
            self.assertLessEqual(s["end_time"], s["window"]["close"])
            prev_end = s["end_time"]
        self.assertEqual(prev_end, res["objective"]["final_end_time"])

    def test_result_independent_of_request_target_order(self):
        raw = reported_scenario()
        reference = plan(raw)
        want_status = {c["id"]: c["status"] for c in reference["classifications"]}
        rng = random.Random(20260922)
        n = len(raw["targets"])
        for _ in range(12):
            perm = list(range(n))
            rng.shuffle(perm)
            res = plan(permute_request(raw, perm))
            self.assertEqual(res["objective"], reference["objective"])
            self.assertEqual(res["canonical_plan"],
                             reference["canonical_plan"])
            self.assertEqual(res["optimal_target_set_count"],
                             reference["optimal_target_set_count"])
            self.assertEqual(
                {c["id"]: c["status"] for c in res["classifications"]},
                want_status,
            )

    def test_renumbered_ids_select_same_business_combination(self):
        raw = reported_scenario()
        id_map = {20: 7, 10: 900, 30: 55,
                  40: -3, 41: -2, 42: -1, 43: 0, 44: 1, 45: 2}
        renumbered = {
            "targets": [
                {**t, "id": id_map[t["id"]]} for t in raw["targets"]
            ],
            "slew": raw["slew"],
        }
        res = plan(renumbered)
        self.assertEqual(res["objective"],
                         {"total_value": 16, "final_end_time": 7})
        # Same business combination {10, 30, 20}, expressed in the new ids.
        self.assertEqual(res["canonical_plan"]["target_ids"], [900, 55, 7])
        self.assertEqual(res["optimal_target_set_count"], 1)
        statuses = {c["id"]: c["status"] for c in res["classifications"]}
        self.assertEqual(
            statuses,
            {7: "required", 900: "required", 55: "required",
             -3: "excluded", -2: "excluded", -1: "excluded",
             0: "excluded", 1: "excluded", 2: "excluded"},
        )

    def test_canonical_lex_smallest_with_slack_prefix(self):
        # Sequence [1,2,3,4] is optimal (value 4, end 9) even though its
        # prefix {1,2,3} ends at 8 while another ordering of that prefix
        # ends at 3.  The canonical plan must still be the lexicographically
        # smallest optimal sequence, not [2,1,3,4].
        raw = make_raw(
            n=4,
            d=[1, 1, 1, 1],
            v=[1, 1, 1, 1],
            w=[[(0, 100)], [(0, 100)], [(0, 100)], [(8, 9)]],
            s0=[5, 0, 0, 0],
            sm=[[0] * 4 for _ in range(4)],
        )
        res = plan(raw)
        self.assertEqual(res["objective"],
                         {"total_value": 4, "final_end_time": 9})
        self.assertEqual(res["canonical_plan"]["target_ids"], [1, 2, 3, 4])
        self.assertMatchesBrute(raw)

    def test_fuzz_against_brute_force(self):
        rng = random.Random(20260919)
        for case in range(120):
            n = rng.randint(2, 7)
            ids = rng.sample(range(-20, 80), n)
            d = [rng.randint(1, 6) for _ in range(n)]
            v = [rng.randint(1, 9) for _ in range(n)]
            w = []
            for i in range(n):
                k = rng.randint(0, 3)
                ws = []
                for _ in range(k):
                    lo = rng.randint(0, 12)
                    hi = lo + rng.randint(0, 8)
                    ws.append((lo, hi))
                w.append(ws)
            s0 = [rng.randint(0, 8) for _ in range(n)]
            sm = [[rng.randint(0, 6) for _ in range(n)] for _ in range(n)]
            raw = make_raw(n, d, v, w, s0, sm, ids=ids)
            self.assertMatchesBrute(raw)

    def test_fuzz_larger_sets_against_brute_force(self):
        # n >= 9 targets (all kept alive by construction) exercise the
        # combinatorial regime where follow-up combinations used to be
        # pruned away.  Brute force is still affordable at these sizes.
        rng = random.Random(20260920)
        for case in range(10):
            n = rng.choice([8, 9])
            ids = rng.sample(range(-30, 90), n)
            d = [rng.randint(1, 4) for _ in range(n)]
            v = [rng.randint(1, 9) for _ in range(n)]
            w = []
            for i in range(n):
                k = rng.randint(1, 2)  # at least one window: target stays alive
                ws = []
                for _ in range(k):
                    lo = rng.randint(0, 10)
                    hi = lo + rng.randint(2, 9)
                    ws.append((lo, hi))
                w.append(ws)
            s0 = [rng.randint(0, 6) for _ in range(n)]
            sm = [[rng.randint(0, 5) for _ in range(n)] for _ in range(n)]
            raw = make_raw(n, d, v, w, s0, sm, ids=ids)
            self.assertMatchesBrute(raw)

    def test_fuzz_permutation_invariance(self):
        # Shuffling the request's target array (and its slew matrices) must
        # not change the objective, the canonical plan, or classifications.
        rng = random.Random(20260921)
        for case in range(12):
            n = rng.randint(2, 9)
            ids = rng.sample(range(-30, 90), n)
            d = [rng.randint(1, 4) for _ in range(n)]
            v = [rng.randint(1, 9) for _ in range(n)]
            w = []
            for i in range(n):
                k = rng.randint(1, 2)
                ws = []
                for _ in range(k):
                    lo = rng.randint(0, 10)
                    hi = lo + rng.randint(2, 9)
                    ws.append((lo, hi))
                w.append(ws)
            s0 = [rng.randint(0, 6) for _ in range(n)]
            sm = [[rng.randint(0, 5) for _ in range(n)] for _ in range(n)]
            raw = make_raw(n, d, v, w, s0, sm, ids=ids)
            reference = plan(raw)
            want_status = {
                c["id"]: c["status"] for c in reference["classifications"]
            }
            for _ in range(3):
                perm = list(range(n))
                rng.shuffle(perm)
                res = plan(permute_request(raw, perm))
                self.assertEqual(res["objective"], reference["objective"])
                self.assertEqual(res["canonical_plan"],
                                 reference["canonical_plan"])
                self.assertEqual(res["optimal_target_set_count"],
                                 reference["optimal_target_set_count"])
                self.assertEqual(
                    {c["id"]: c["status"] for c in res["classifications"]},
                    want_status,
                )

    # ------------------------------------------------------------- invalid

    def _expect_error(self, raw):
        with self.assertRaises(PlanError):
            plan(raw)

    def test_invalid_inputs_rejected_wholesale(self):
        base = make_raw(
            2, [1, 1], [1, 1], [[(0, 5)], [(0, 5)]], [0, 0],
            [[0, 0], [0, 0]],
        )
        bad_count = {**base, "targets": base["targets"][:1]}
        self._expect_error(bad_count)
        too_many = make_raw(
            19, [1] * 19, [1] * 19, [[(0, 5)]] * 19, [0] * 19,
            [[0] * 19 for _ in range(19)],
        )
        self._expect_error(too_many)

        dup = {
            "targets": [
                {"id": 1, "duration": 1, "value": 1, "windows": [{"open": 0, "close": 5}]},
                {"id": 1, "duration": 1, "value": 1, "windows": [{"open": 0, "close": 5}]},
            ],
            "slew": {"from_night_start": [0, 0], "between_targets": [[0, 0], [0, 0]]},
        }
        self._expect_error(dup)

        cases = [
            lambda r: r["targets"][0].__setitem__("duration", 0),
            lambda r: r["targets"][0].__setitem__("duration", -2),
            lambda r: r["targets"][1].__setitem__("value", 0),
            lambda r: r["targets"][0]["windows"][0].__setitem__("open", -1),
            lambda r: r["targets"][0]["windows"][0].__setitem__("close", -2),
            lambda r: r["targets"][0]["windows"].extend(
                [{"open": 6, "close": 9}, {"open": 10, "close": 12},
                 {"open": 13, "close": 15}]),
            lambda r: r["slew"].__setitem__("from_night_start", [0]),
            lambda r: r["slew"].__setitem__("between_targets", [[0, 0]]),
            lambda r: r["slew"]["between_targets"][0].__setitem__(1, -1),
            lambda r: r["targets"][0].__setitem__("duration", 1.5),
            lambda r: r["targets"][0].__setitem__("duration", True),
        ]
        for mutate in cases:
            raw = {
                "targets": [dict(t, windows=list(t["windows"])) for t in base["targets"]],
                "slew": {
                    "from_night_start": list(base["slew"]["from_night_start"]),
                    "between_targets": [list(row) for row in base["slew"]["between_targets"]],
                },
            }
            # deep copy windows
            raw["targets"] = [
                {**t, "windows": [dict(x) for x in t["windows"]]}
                for t in raw["targets"]
            ]
            mutate(raw)
            self._expect_error(raw)

        self._expect_error({"targets": [], "slew": {}})
        self._expect_error([])

    def test_four_windows_rejected_but_three_accepted(self):
        raw = make_raw(
            2, [1, 1], [1, 1], [[(0, 5)], [(0, 5)]], [0, 0],
            [[0, 0], [0, 0]],
        )
        raw["targets"][0]["windows"].extend(
            [{"open": 6, "close": 9}, {"open": 10, "close": 12}]
        )
        # 3 windows: accepted
        plan(raw)
        raw["targets"][0]["windows"].append({"open": 13, "close": 15})
        # 4 windows: rejected
        self._expect_error(raw)


if __name__ == "__main__":
    unittest.main()
