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
end time.  Within each mask a sound Pareto frontier drops only states that
reach every target no later than another ordering of the *same* mask;
pruning across distinct masks is never valid because values are positive.
Optimality witnesses for lexicographic reconstruction are then found by an
on-demand dp-tight DFS that falls back to a full reverse closure if the
optimal-plan region turns out to be dense.  Pure standard library.
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
    # Single-window targets admit a pure arithmetic feasibility check:
    # (open, latest-ready).  Others fall back to the window scan/cache.
    single = [
        (wins[k][0][0], wins[k][0][1] - dur[k]) if len(wins[k]) == 1 else None
        for k in range(m)
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
    # Incremental within-mask frontier summary, fed on every dp write:
    # while a mask has a single non-dominated state it is cached in
    # hint_last/hint_end and needs no member scan at all; once two
    # incomparable states appear the mask is flagged multi and its frontier
    # is rebuilt authoritatively from dp.  Hints are only an optimization --
    # the multi-path recompute never trusts them.
    hint_last = [-1] * size
    hint_end = [-1] * size
    multi = bytearray(size)
    for i in range(m):
        if first_end[i] != UNREACH:
            p = (1 << i) * m + i
            dp[p] = first_end[i]
            hint_last[1 << i] = i
            hint_end[1 << i] = first_end[i]

    # dmax[a][b] = max over all targets j of (slew[b][j] - slew[a][j]).
    # A state ending in b with end e_b reaches every target no later than a
    # state ending in a with end e_a iff dmax[a][b] <= e_a - e_b.  Using all
    # j (rather than only not-yet-used targets) only makes the test a
    # sufficient -- never a necessary -- condition, so it stays exact.
    dmax = [[0] * m for _ in range(m)]
    for a in range(m):
        ra = sm[a]
        for b in range(m):
            rb = sm[b]
            d = 0
            for j in range(m):
                diff = rb[j] - ra[j]
                if diff > d:
                    d = diff
            dmax[a][b] = d

    # Hot transition loop, kept flat (local aliases, tuple windows).
    _wins, _dur, _sm = wins, dur, sm
    _lsb, _off = lsb_index, offset_k
    _latest = latest_ready
    _single = single
    _dmax = dmax
    _unreach = UNREACH
    # Memoize earliest start per (target, slew-finish time): the same ready
    # time recurs across many (mask, predecessor) pairs; feasibility of a
    # target depends only on its windows and the ready time. -2 = infeasible.
    ready_cache = [dict() for _ in range(m)]
    cache_get = [c.get for c in ready_cache]
    scratch = [0] * m  # per-mask minimum slew-ready time per next target
    INF = 1 << 62

    def _publish(nmask, nxt, new_end):
        """Fold a dp state into the target mask's frontier summary.

        Only used by the (rare) multi-frontier path; the hot path inlines
        this to avoid a call per transition.
        """
        if multi[nmask]:
            return
        hl = hint_last[nmask]
        if hl < 0:
            hint_last[nmask] = nxt
            hint_end[nmask] = new_end
        elif hl == nxt:
            hint_end[nmask] = new_end
        elif _dmax[nxt][hl] <= new_end - hint_end[nmask]:
            pass  # cached state reaches every target no later
        elif _dmax[hl][nxt] <= hint_end[nmask] - new_end:
            hint_last[nmask] = nxt
            hint_end[nmask] = new_end
        else:
            multi[nmask] = 1  # rebuild later

    for mask in range(1, size):
        base = mask * m
        cand = full ^ mask
        if not multi[mask]:
            f_last = hint_last[mask]
            if f_last < 0:
                continue
            f_end = hint_end[mask]
            # Tight path: a single non-dominated ordering of this mask.
            row = _sm[f_last]
            c = cand
            while c:
                b = c & -c
                c ^= b
                nxt = _lsb[b]
                best_ready = f_end + row[nxt]
                sw = _single[nxt]
                if sw is not None:
                    lo0, cap = sw
                    if best_ready > cap:
                        continue
                    st = best_ready if best_ready > lo0 else lo0
                else:
                    if best_ready > _latest[nxt]:
                        continue
                    st = cache_get[nxt](best_ready)
                    if st is None:
                        st = -2
                        for lo, hi in _wins[nxt]:
                            t = lo if best_ready <= lo else best_ready
                            if t + _dur[nxt] <= hi and (st < 0 or t < st):
                                st = t
                        ready_cache[nxt][best_ready] = st
                if st >= 0:
                    p = base + _off[nxt]
                    new_end = st + _dur[nxt]
                    old_end = dp[p]
                    if old_end == _unreach or new_end < old_end:
                        dp[p] = new_end
                        nmask = mask | b
                        if not multi[nmask]:
                            hl = hint_last[nmask]
                            if hl < 0:
                                hint_last[nmask] = nxt
                                hint_end[nmask] = new_end
                            elif hl == nxt:
                                hint_end[nmask] = new_end
                            elif _dmax[nxt][hl] <= new_end - hint_end[nmask]:
                                pass  # cached state reaches every target no later
                            elif _dmax[hl][nxt] <= hint_end[nmask] - new_end:
                                hint_last[nmask] = nxt
                                hint_end[nmask] = new_end
                            else:
                                multi[nmask] = 1  # rebuild later
            continue

        # General path: rebuild the sound within-mask Pareto frontier from
        # dp.  (e2, l2) makes (end, last) redundant when it reaches every
        # target no later; both states have observed exactly `mask`.
        # Cross-mask comparison would conflate distinct target sets (values
        # are positive) and is never done.
        retained = []
        members = mask
        while members:
            b = members & -members
            members ^= b
            last = _lsb[b]
            end = dp[base + last]
            if end == _unreach:
                continue
            dominated = False
            i = 0
            while i < len(retained):
                e2, l2 = retained[i]
                if _dmax[last][l2] <= end - e2:
                    dominated = True
                    break
                if _dmax[l2][last] <= e2 - end:
                    retained[i] = retained[-1]
                    retained.pop()
                    continue
                i += 1
            if not dominated:
                retained.append((end, last))
        if not retained:
            continue

        # Min slew-ready time to each still-unobserved target over the
        # frontier.  Only the earliest predecessor can produce the earliest
        # end of state (mask | nxt, nxt): feasibility is monotone in ready.
        c = cand
        while c:
            b = c & -c
            c ^= b
            scratch[_lsb[b]] = INF
        for end, last in retained:
            row = _sm[last]
            c = cand
            while c:
                b = c & -c
                c ^= b
                nxt = _lsb[b]
                ready = end + row[nxt]
                if ready < scratch[nxt]:
                    scratch[nxt] = ready

        c = cand
        while c:
            b = c & -c
            c ^= b
            nxt = _lsb[b]
            best_ready = scratch[nxt]
            sw = _single[nxt]
            if sw is not None:
                lo0, cap = sw
                if best_ready > cap:
                    continue
                st = best_ready if best_ready > lo0 else lo0
            else:
                if best_ready > _latest[nxt]:
                    continue
                st = cache_get[nxt](best_ready)
                if st is None:
                    # Earliest feasible start across at most three windows.
                    st = -2
                    for lo, hi in _wins[nxt]:
                        t = lo if best_ready <= lo else best_ready
                        if t + _dur[nxt] <= hi and (st < 0 or t < st):
                            st = t
                    ready_cache[nxt][best_ready] = st
            if st >= 0:
                p = base + _off[nxt]  # (mask | b) * m + nxt
                new_end = st + _dur[nxt]
                old_end = dp[p]
                if old_end == _unreach or new_end < old_end:
                    dp[p] = new_end
                    _publish(mask | b, nxt, new_end)

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

    # On-demand optimality witness test: can state (mask, last) be extended,
    # using only dp-tight transitions, to an optimal mask ending at
    # best_end?  Only states visited by the lexicographic reconstruction are
    # ever queried; in dense all-tie cases a witness path is found after a
    # handful of steps.  If proving the answers starts exploring most of the
    # state space, one flat full reverse closure is built instead and all
    # further lookups read it.  Single-window targets use pure arithmetic.
    finish_yes = bytearray(size * m)
    in_optimal = optimal_masks.__contains__
    closure_done = False
    expanded = 0
    EXPAND_BUDGET = 256  # witness paths are at most n steps each

    def _tight_start(last, nxt, end):
        ready = end + sm[last][nxt]
        sw = single[nxt]
        if sw is not None:
            lo, cap = sw
            if ready > cap:
                return -1
            return ready if ready > lo else lo
        if ready > latest_ready[nxt]:
            return -1
        st = ready_cache[nxt].get(ready)
        if st is None:
            st = -2
            for lo, hi in wins[nxt]:
                t = ready if ready > lo else lo
                if t + dur[nxt] <= hi and (st < 0 or t < st):
                    st = t
            ready_cache[nxt][ready] = st
        return st

    def _build_full_closure():
        # Stack-based reverse closure: seed every optimal-mask state at
        # best_end, walk predecessor edges whose dp arrival is tight.  Cheap
        # when the good region is sparse; marks dominated intermediates too.
        good = finish_yes
        stack = []
        for om in optimal_masks:
            if om == 0:
                continue
            base = om * m
            x = om
            while x:
                b = x & -x
                x ^= b
                last = lsb_index[b]
                if dp[base + last] == best_end:
                    p = base + last
                    if not good[p]:
                        good[p] = 1
                        stack.append((om, last))
        while stack:
            fmask, last = stack.pop()
            prev_mask = fmask ^ (1 << last)
            if prev_mask == 0:
                continue
            target_start = dp[fmask * m + last] - dur[last]
            base = prev_mask * m
            x = prev_mask
            while x:
                b = x & -x
                x ^= b
                prev = lsb_index[b]
                prev_end = dp[base + prev]
                if prev_end == UNREACH:
                    continue
                st = _tight_start(prev, last, prev_end)
                if st == target_start:
                    p = base + prev
                    if not good[p]:
                        good[p] = 1
                        stack.append((prev_mask, prev))

    def can_finish(mask, last):
        nonlocal closure_done, expanded
        p0 = mask * m + last
        if finish_yes[p0]:
            return True
        if in_optimal(mask) and dp[p0] == best_end:
            finish_yes[p0] = 1
            return True
        if closure_done:
            return False  # full closure is authoritative
        # Explicit-stack DFS over the strictly-growing mask DAG.  Frames are
        # (mask, last, end, successor-bits-left).
        stack = [(mask, last, dp[p0], full ^ mask)]
        trail = [p0]
        answer = False
        while stack:
            fm, fl, fend, c = stack[-1]
            if c:
                b = c & -c
                stack[-1] = (fm, fl, fend, c ^ b)
                nxt = lsb_index[b]
                st = _tight_start(fl, nxt, fend)
                if st < 0:
                    continue
                nmask = fm | b
                np = nmask * m + nxt
                if st + dur[nxt] != dp[np]:
                    continue  # not dp-tight: a better arrival exists
                if finish_yes[np]:
                    answer = True
                    break
                if in_optimal(nmask) and dp[np] == best_end:
                    finish_yes[np] = 1
                    answer = True
                    break
                expanded += 1
                if expanded > EXPAND_BUDGET:
                    _build_full_closure()
                    closure_done = True
                    return bool(finish_yes[p0])
                stack.append((nmask, nxt, dp[np], full ^ nmask))
                trail.append(np)
            else:
                stack.pop()
                trail.pop()
        if answer:
            for p in trail:
                finish_yes[p] = 1
        return answer

    # Membership of every target across all criteria-1+2 optimal plans.
    ever_in = 0
    ever_out = 0
    for om in optimal_masks:
        ever_in |= om
        ever_out |= full ^ om

    class _Witness:
        """good-state lookup backed by the memoized witness DFS/closure."""

        __slots__ = ()

        def __getitem__(self, p):
            return 1 if can_finish(p // m, p % m) else 0

    canonical_steps, canonical_ids = _canonical(
        m, ids, alive, dur, wins, s0, sm, dp, _Witness(), optimal_masks
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


def _canonical(m, ids, alive, dur, wins, s0, sm, dp, can_witness, optimal_masks):
    """Greedy reconstruction of the lexicographically smallest optimal plan.

    At each position take the smallest id whose next state admits an
    optimality witness (a dp-tight continuation to an optimal mask); the
    earliest-start timeline is then forced (dp times), so each sequence has
    exactly one reported schedule.
    """
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
            end = dp[new_mask * m + k]
            if end == UNREACH or not can_witness[new_mask * m + k]:
                continue
            ready = s0[k] if last == -1 else prev_end + sm[last][k]
            st, _ = earliest_with_window(k, ready)
            if st is None or st + dur[k] != end:
                continue
            chosen = k
            chosen_ready = ready
            chosen_start = st
            chosen_end = end
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

    if optimal_masks and mask not in optimal_masks:
        # Defensive: should be impossible by construction of the closure.
        raise RuntimeError("internal: reconstructed plan is not optimal")

    return steps, out_ids
