# Night Planner — transient-target observation scheduler

Pure back-end service for planning a single transient-tracking night.
Given 2–18 unique targets with exposure durations, science values, short
visibility windows and slew times, it finds the **exact** optimum — no
priority/nearest-neighbor heuristic — and tells the operator whether each
target is a robust pick or merely an artifact of the tie-breaking rules.

* Python 3 standard library only — no third-party packages, **no general
  optimization solver**, no front end, no online service calls.
* Exact subset dynamic programming (n = 18 runs in a few seconds).
* Docker + Docker Compose, container health check, configurable host port.

## Run

```bash
docker compose up --build
```

Publish on a different host port via environment variable:

```bash
API_HOST_PORT=9090 docker compose up --build
```

Other variables (all optional):

| Variable             | Default   | Meaning                                   |
| -------------------- | --------- | ----------------------------------------- |
| `API_HOST_PORT`      | `8080`    | port published on the host                |
| `API_PORT`           | `8080`    | port the service listens on (in container)|
| `API_HOST`           | `0.0.0.0` | bind address                              |
| `API_MAX_BODY_BYTES` | `1048576` | maximum accepted request body (1 MiB)     |

Health check:

```bash
curl -s http://localhost:8080/health
# {"status":"ok","uptime_seconds":12.3}
```

Run locally without Docker (also no dependencies):

```bash
python -m app.server
```

Run the tests (incl. a brute-force cross-check on randomized small cases):

```bash
python -m unittest discover -s tests
```

## API

### `POST /plan`

All times are integer seconds measured from the night origin (`t = 0`).

Request fields:

* `targets` — **2 to 18** objects, unique integer `id`:
  * `duration` — positive integer, uninterruptible exposure length;
  * `value` — positive integer science value;
  * `windows` — **0 to 3** closed integer-second windows
    `{"open", "close"}` (`0 <= open <= close`). An exposure starting at `t`
    is legal in a window iff `open <= t` and `t + duration <= close`
    (boundary inclusive; zero windows = target never visible).
* `slew`:
  * `from_night_start` — n non-negative integers, slew time from the night
    origin directly to each target;
  * `between_targets` — n × n matrix of non-negative integers,
    `between_targets[i][j]` being the slew from target i to target j.
    No triangle inequality is assumed (reaching a target via another one
    may be faster).

Rules: every target at most once; exposures cannot be interrupted; the
next exposure may only start after the slew finishes **and** inside a
legal window. Start times are integers.

Optimization criteria, applied **in this strict order**:

1. maximize total science value;
2. minimize the final exposure end time (the empty plan ends at `t = 0`);
3. minimize the lexicographic order of the sequence of target ids.

`classifications` describes each target's membership across **all plans
that are jointly optimal under criteria 1 and 2** (before the lexicographic
rule 3 picks one canonical sequence):

* `required` — present in every optimal plan;
* `optional` — present in some optimal plan, absent in another;
* `excluded` — present in none.

Example (`examples/request.json`): a greedy choice of the 100-point target
loses to the 120-point combination:

```bash
curl -s -X POST http://localhost:8080/plan \
  -H 'Content-Type: application/json' \
  --data @examples/request.json
```

```json
{
  "canonical_plan": {
    "target_ids": [2, 3],
    "steps": [
      {"order": 1, "id": 2,
       "slew": {"from": "NIGHT_START", "seconds": 0, "finish_time": 0},
       "start_time": 0, "end_time": 5, "exposure_seconds": 5,
       "window": {"open": 0, "close": 6}},
      {"order": 2, "id": 3,
       "slew": {"from": 2, "seconds": 1, "finish_time": 6},
       "start_time": 6, "end_time": 11, "exposure_seconds": 5,
       "window": {"open": 6, "close": 12}}
    ]
  },
  "objective": {"total_value": 120, "final_end_time": 11},
  "optimal_target_set_count": 1,
  "empty_plan": false,
  "classifications": [
    {"id": 1, "status": "excluded"},
    {"id": 2, "status": "required"},
    {"id": 3, "status": "required"}
  ],
  "target_count": 3
}
```

Each step is independently re-checkable:

* `slew.finish_time` = previous step's `end_time` + `slew.seconds`
  (or `slew.seconds` from the night start);
* `start_time >= slew.finish_time`;
* `window.open <= start_time` and
  `end_time == start_time + exposure_seconds <= window.close`.

### Zero-value conclusions

When no target can be observed (all windows empty/infeasible, or no slew can
arrive in time), or the optimum is the empty plan, the service returns the
fully recomputable result

```json
{
  "canonical_plan": {"target_ids": [], "steps": []},
  "objective": {"total_value": 0, "final_end_time": 0},
  "empty_plan": true,
  ...
}
```

with every target `excluded`.

### Invalid input

Any malformed request is **rejected as a whole** with `400 Bad Request`
and `{"error": "invalid_request", "detail": "..."}` — no partial plan is
returned. Malformed JSON yields `{"error": "invalid_json"}`.

## How the algorithm works

`app/planner.py`:

1. Drop targets whose exposure fits in none of their windows.
2. Subset DP over `(observed-mask, last-target)` storing the earliest
   achievable exposure end time. Each transition appends one target:
   slew after the previous end, then earliest legal start in any window.
3. Scan all reachable masks to select those maximizing value and, among
   them, minimizing end time.
4. A backward pass over subsets of the optimal masks computes, for every
   `(mask, last-target)` state, the latest exposure end time that can
   still be completed into an optimal plan (completability is monotone in
   the arrival time, so one deadline per state is exact). Membership
   across the optimal masks gives required / optional / excluded.
5. Greedy smallest-id walk through states whose earliest-start arrival
   meets the deadline reconstructs the unique lexicographically smallest
   plan, with a fully determined timeline (earliest legal start at every
   step).
