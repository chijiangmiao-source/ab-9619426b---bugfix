"""Exact scheduler for the transient-tracking night.

All times are integer seconds measured from the night origin (t = 0).

Rules
-----
* Target ``i`` has positive integer exposure ``duration`` and science
  ``value``, plus 0..3 closed integer-second windows ``[open, close]``
  (zero windows means the target is never visible).
* Slewing takes a non-negative integer time: ``slew_night[i]`` from the
  night origin, ``slew[i][j]`` between targets.  No triangle inequality is
  assumed: reaching ``j`` via another target may be faster than directly.
* An exposure is uninterruptible.  It may start at integer ``t`` only when
  the slew has finished (``t >= ready``) and some window contains the whole
  exposure (``open <= t`` and ``t + duration <= close``).
* Each target is observed at most once.

Tie-breaking, in strict order
-----------------------------
1. maximize total science value;
2. minimize the final exposure end time (the empty plan ends at t = 0);
3. minimize the lexicographic order of the sequence of target ids.

The response reports the canonical plan (all three criteria) *and* every
target's membership across all plans tied on criteria 1+2:
``required`` / ``optional`` / ``excluded``.

Algorithm: subset DP over ``(mask, last)`` storing the earliest achievable
end time; masks maximizing value and then minimizing end time are the
optimal target sets; a backward ``latest[mask, last]`` pass computes the
newest exposure-end time from which each state can still be completed into
a criteria-1+2 optimal plan; the canonical sequence is recovered by greedy
smallest-id extension through states whose earliest-start arrival does not
exceed that deadline.  Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_TARGETS = 2
MAX_TARGETS = 18
MAX_WINDOWS = 3
UNREACH = -1


class PlanError(ValueError):
    """Raised for any invalid request; such requests are rejected wholesale."""


@dataclass(frozen=True)
class Window:
    open: int
    close: int  # latest permitted *end* of an exposure


# ---------------------------------------------------------------- validation


def _as_int(value, what: str) -> int:
    # bool is a subclass of int -- reject it for strict typing.
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{what} must be an integer")
    return value


def _as_pos(value, what: str) -> int:
    value = _as_int(value, what)
    if value <= 0:
        raise PlanError(f"{what} must be a positive integer")
    return value


def _as_nonneg(value, what: str) -> int:
    value = _as_int(value, what)
    if value < 0:
        raise PlanError(f"{what} must be a non-negative integer")
    return value


def _validate(raw: dict):
    if not isinstance(raw, dict):
        raise PlanError("request body must be a JSON object")

    raw_targets = raw.get("targets")
    if not isinstance(raw_targets, list):
        raise PlanError("'targets' must be a list")
    n = len(raw_targets)
    if not (MIN_TARGETS <= n <= MAX_TARGETS):
        raise PlanError(
            f"number of targets must be between {MIN_TARGETS} and {MAX_TARGETS}"
        )

    ids = []
    durations = []
    values = []
    windows = []
    seen_ids = set()
    for idx, item in enumerate(raw_targets):
        where = f"targets[{idx}]"
        if not isinstance(item, dict):
            raise PlanError(f"{where} must be an object")

        tid = _as_int(item.get("id"), f"{where}.id")
        if tid in seen_ids:
            raise PlanError(f"duplicate target id: {tid}")
        seen_ids.add(tid)
        ids.append(tid)

        durations.append(_as_pos(item.get("duration"), f"{where}.duration"))
        values.append(_as_pos(item.get("value"), f"{where}.value"))

        # "At most three windows" -- zero windows means the target is never
        # visible and can never be observed (it stays a valid request).
        raw_wins = item.get("windows", [])
        if not isinstance(raw_wins, list):
            raise PlanError(f"{where}.windows must be a list")
        if len(raw_wins) > MAX_WINDOWS:
            raise PlanError(f"{where} has more than {MAX_WINDOWS} windows")
        parsed = []
        for w_idx, w in enumerate(raw_wins):
            wwhere = f"{where}.windows[{w_idx}]"
            if not isinstance(w, dict):
                raise PlanError(f"{wwhere} must be an object")
            lo = _as_int(w.get("open"), f"{wwhere}.open")
            hi = _as_int(w.get("close"), f"{wwhere}.close")
            if lo < 0:
                raise PlanError(f"{wwhere}.open must be non-negative")
            if hi < lo:
                raise PlanError(f"{wwhere}.close must be >= open")
            parsed.append(Window(lo, hi))
        # Stable order for deterministic evidence; overlaps are harmless
        # because earliest-start scans every window.
        parsed.sort(key=lambda w: (w.open, w.close))
        windows.append(parsed)

    slew_night, slew = _validate_slew(raw, n)
    return ids, durations, values, windows, slew_night, slew


def _validate_slew(raw: dict, n: int):
    slew_obj = raw.get("slew")
    if not isinstance(slew_obj, dict):
        raise PlanError("'slew' must be an object")

    night = slew_obj.get("from_night_start")
    if not isinstance(night, list) or len(night) != n:
        raise PlanError(
            "slew.from_night_start must be a list with one entry per target"
        )
    slew_night = [
        _as_nonneg(x, f"slew.from_night_start[{i}]") for i, x in enumerate(night)
    ]

    matrix = slew_obj.get("between_targets")
    if not isinstance(matrix, list) or len(matrix) != n:
        raise PlanError("slew.between_targets must be an n x n matrix")
    slew = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != n:
            raise PlanError(f"slew.between_targets[{i}] must have {n} entries")
        slew.append(
            [
                _as_nonneg(x, f"slew.between_targets[{i}][{j}]")
                for j, x in enumerate(row)
            ]
        )
    return slew_night, slew


# ----------------------------------------------------------------- algorithm


def _earliest_start(windows, ready: int, duration: int):
    """Earliest integer start >= ready with the full exposure in a window."""
    best = None
    for w in windows:
        start = ready if ready > w.open else w.open
        if start + duration <= w.close and (best is None or start < best):
            best = start
    return best


def plan(raw_request: dict) -> dict:
    ids, durations, values, windows, slew_night, slew = _validate(raw_request)
    n = len(ids)

    # A target whose exposure fits in none of its windows can never be
    # observed under any slew history; drop it from the combinatorial part.
    alive = [
        i
        for i in range(n)
        if any(w.close - w.open >= durations[i] for w in windows[i])
    ]
    alive_set = set(alive)
    compressed = {orig: k for k, orig in enumerate(alive)}
    m = len(alive)

    # Compressed-index arrays. Windows become flat tuples for speed:
    # wins[k] = ((open, close), ...)
    dur = [durations[i] for i in alive]
    val = [values[i] for i in alive]
    wins = [tuple((w.open, w.close) for w in windows[i]) for i in alive]
    s0 = [slew_night[i] for i in alive]
    sm = [[slew[i][j] for j in alive] for i in alive]

    size = 1 << m
    full = size - 1

    # Index of the single set bit of each power-of-two mask (else -1).
    lsb_index = [-1] * size
    for k in range(m):
        lsb_index[1 << k] = k
    # Flat-state offset of appending target k: (1 << k) * m + k.
    offset_k = [(1 << k) * m + k for k in range(m)]
    # Latest slew-completion time that can still start target k: beyond
    # max(close) - duration no window can contain the exposure. (Alive
    # targets have at least one window fitting their duration.)
    latest_ready = [
        max(hi for lo, hi in wins[k]) - dur[k] for k in range(m)
    ]

    # First-observation seeds (slew directly from the night origin).
    first_end = [UNREACH] * m
    for i in range(m):
        st = _earliest_start(windows[alive[i]], s0[i], dur[i])
        if st is not None:
            first_end[i] = st + dur[i]

    # subset_value[mask]
    subset_value = [0] * size
    for mask in range(1, size):
        bit = mask & -mask
        subset_value[mask] = subset_value[mask ^ bit] + val[lsb_index[bit]]

    # dp[mask * m + last] = earliest end of an ordering of exactly `mask`
    # ending in `last`.  Iterating masks in numeric order is a topological
    # order since every transition adds one bit (mask | bit > mask).
    dp = [UNREACH] * (size * m)
    for i in range(m):
        if first_end[i] != UNREACH:
            dp[(1 << i) * m + i] = first_end[i]

    # Hot transition loop, kept flat (local aliases, tuple windows).
    _wins, _dur, _sm = wins, dur, sm
    _lsb, _off = lsb_index, offset_k
    _latest = latest_ready
    # Memoize earliest start per (target, slew-finish time): the same ready
    # time recurs across many predecessor masks; feasible start depends
    # only on the target's windows and ready. -2 caches "infeasible".
    ready_cache = [dict() for _ in range(m)]
    for mask in range(1, size):
        base = mask * m
        remaining = full ^ mask
        members = mask
        while members:
            b = members & -members
            members ^= b
            last = _lsb[b]
            end = dp[base + last]
            if end == UNREACH:
                continue
            row = _sm[last]
            cand = remaining
            while cand:
                cb = cand & -cand
                cand ^= cb
                nxt = _lsb[cb]
                ready = end + row[nxt]
                if ready > _latest[nxt]:
                    continue
                st = ready_cache[nxt].get(ready)
                if st is None:
                    # Earliest feasible start across at most three windows.
                    st = -2
                    for lo, hi in _wins[nxt]:
                        t = lo if ready <= lo else ready
                        if t + _dur[nxt] <= hi and (st < 0 or t < st):
                            st = t
                    ready_cache[nxt][ready] = st
                if st >= 0:
                    p = base + _off[nxt]  # (mask | cb) * m + nxt
                    old = dp[p]
                    if old == UNREACH or st + _dur[nxt] < old:
                        dp[p] = st + _dur[nxt]

    # Criteria 1+2 over all reachable masks (the empty mask always is).
    best_value = 0
    best_end = 0
    optimal_masks = {0}
    for mask in range(1, size):
        e = UNREACH
        base = mask * m
        x = mask
        while x:
            b = x & -x
            x ^= b
            v = dp[base + lsb_index[b]]
            if v != UNREACH and (e == UNREACH or v < e):
                e = v
        if e == UNREACH:
            continue
        v = subset_value[mask]
        if v > best_value or (v == best_value and e < best_end):
            best_value = v
            best_end = e
            optimal_masks = {mask}
        elif v == best_value and e == best_end:
            optimal_masks.add(mask)

    # Membership of every target across all criteria-1+2 optimal plans.
    ever_in = 0
    ever_out = 0
    for om in optimal_masks:
        ever_in |= om
        ever_out |= full ^ om

    canonical_steps, canonical_ids = _canonical(
        m, ids, alive, dur, wins, s0, sm, dp, lsb_index, offset_k,
        subset_value, optimal_masks, best_value, best_end,
    )

    classifications = []
    for idx in range(n):
        if idx in alive_set:
            k = compressed[idx]
            bit = 1 << k
            in_any = bool(ever_in & bit)
            out_any = bool(ever_out & bit)
            status = (
                "required"
                if in_any and not out_any
                else "optional"
                if in_any
                else "excluded"
            )
        else:
            status = "excluded"  # never visible: in no optimal plan
        classifications.append({"id": ids[idx], "status": status})

    return {
        "canonical_plan": {
            "target_ids": canonical_ids,
            "steps": canonical_steps,
        },
        "objective": {
            "total_value": best_value,
            "final_end_time": best_end,
        },
        "optimal_target_set_count": len(optimal_masks),
        "empty_plan": best_value == 0,
        "classifications": classifications,
        "target_count": n,
    }


def _canonical(m, ids, alive, dur, wins, s0, sm, dp, lsb, offset_k,
               subset_value, optimal_masks, best_value, best_end):
    """Greedy reconstruction of the lexicographically smallest optimal plan.

    ``latest[mask * m + last]`` is the newest exposure-end time at which
    state ``(mask, last)`` can still be completed into a criteria-1+2
    optimal plan (total value ``best_value``, final end ``best_end``), or
    -1 when no such completion exists.  Completability is monotone in the
    arrival time -- any schedule feasible at ``t`` is feasible earlier --
    so this single deadline per state decides exactly which prefixes
    extend to an optimal plan, including optimal plans whose prefixes are
    not themselves dp-earliest.

    At each position take the smallest id whose earliest-start arrival
    does not exceed the next state's deadline; the earliest-start timeline
    is then forced, so each sequence has exactly one reported schedule.
    """
    size = 1 << m
    full = size - 1

    # Masks that can appear on an optimal plan: subsets of an optimal mask.
    on_optimal_path = bytearray(size)
    for om in optimal_masks:
        sub = om
        while True:
            on_optimal_path[sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & om

    # Base: an optimal mask ending at best_end is a complete optimal plan.
    # Its value is already best_value, so it can never be extended; states
    # there are final.  (best_end is the minimum end among maximum-value
    # plans, so a value-optimal mask can only be reached at best_end.)
    latest = [-1] * (size * m)
    for om in optimal_masks:
        base = om * m
        x = om
        while x:
            b = x & -x
            x ^= b
            last = lsb[b]
            if dp[base + last] == best_end:
                latest[base + last] = best_end

    # Backward pass: masks in decreasing numeric order, so every strict
    # superset (numerically larger) is finalized before its subsets.
    # spans[k] = feasible-start intervals [open, close - duration] of k.
    spans = [tuple((lo, hi - dur[k]) for lo, hi in wins[k]) for k in range(m)]
    for mask in range(size - 1, -1, -1):
        if not on_optimal_path[mask] or subset_value[mask] >= best_value:
            continue
        base = mask * m
        # Usable final targets of this mask: the walk can only arrive at a
        # state whose earliest end does not already exceed best_end.
        lasts = []
        x = mask
        while x:
            b = x & -x
            x ^= b
            last = lsb[b]
            e = dp[base + last]
            if e != UNREACH and e <= best_end:
                lasts.append(last)
        if not lasts:
            continue
        # Newest feasible start per appendable target k, ending by the
        # successor's deadline; independent of `last`, so computed once.
        cand_k = []
        cand_start = []
        cand = full ^ mask
        while cand:
            cb = cand & -cand
            cand ^= cb
            if not on_optimal_path[mask | cb]:
                continue
            k = lsb[cb]
            ln = latest[base + offset_k[k]]  # latest[(mask|cb) * m + k]
            if ln < 0:
                continue
            limit = ln - dur[k]
            start = -1
            for lo, hi in spans[k]:
                t = limit if limit < hi else hi
                if t >= lo and t > start:
                    start = t
            if start >= 0:
                cand_k.append(k)
                cand_start.append(start)
        if not cand_k:
            continue
        for last in lasts:
            row = sm[last]
            best_depart = -1
            for i in range(len(cand_k)):
                # Departing (mask, last) at start - slew arrives in time.
                depart = cand_start[i] - row[cand_k[i]]
                if depart > best_depart:
                    best_depart = depart
            latest[base + last] = best_depart

    # Compressed indices ordered by user-facing target id.
    order = sorted(range(m), key=lambda k: ids[alive[k]])

    def earliest_with_window(k, ready):
        """(start, (open, close)) of the window forcing the earliest start."""
        best = None
        best_w = None
        for win in wins[k]:
            lo, hi = win
            t = ready if ready > lo else lo
            if t + dur[k] <= hi and (best is None or t < best):
                best, best_w = t, win
        return best, best_w

    mask = 0
    last = -1
    prev_end = 0
    steps = []
    out_ids = []

    while True:
        chosen = -1
        for k in order:
            bit = 1 << k
            if mask & bit:
                continue
            new_mask = mask | bit
            if not on_optimal_path[new_mask]:
                continue
            deadline = latest[new_mask * m + k]
            if deadline < 0:
                continue
            ready = s0[k] if last == -1 else prev_end + sm[last][k]
            st, _ = earliest_with_window(k, ready)
            if st is None or st + dur[k] > deadline:
                continue
            chosen = k
            chosen_ready = ready
            chosen_start = st
            chosen_end = st + dur[k]
            break
        if chosen == -1:
            break

        mask |= 1 << chosen
        _, win = earliest_with_window(chosen, chosen_ready)
        step = {
            "order": len(steps) + 1,
            "id": ids[alive[chosen]],
            "slew": {
                "from": "NIGHT_START" if last == -1 else ids[alive[last]],
                "seconds": s0[chosen] if last == -1 else sm[last][chosen],
                "finish_time": chosen_ready,
            },
            "start_time": chosen_start,
            "end_time": chosen_end,
            "exposure_seconds": dur[chosen],
            "window": {"open": win[0], "close": win[1]},
        }
        steps.append(step)
        out_ids.append(ids[alive[chosen]])
        last = chosen
        prev_end = chosen_end

    if subset_value[mask] != best_value or prev_end != best_end:
        # Defensive: should be impossible by construction of the deadlines.
        raise RuntimeError("internal: reconstructed plan is not optimal")

    return steps, out_ids
