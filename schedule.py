from ortools.sat.python import cp_model

def build_allowed_start_domain(horizon, sleep_start_min, sleep_end_min):
    """
    sleep window repeats daily.
    sleep_start_min/sleep_end_min are minutes in day [0, 1440).
    Assume sleep_start < sleep_end (e.g., 30, 510). If you want跨午夜，也能再擴充。
    """
    allowed = []
    day = 0
    while day * 1440 < horizon:
        base = day * 1440
        # allowed: [base, base+sleep_start) and [base+sleep_end, base+1440)
        a1_s, a1_e = base, min(base + sleep_start_min, horizon)
        a2_s, a2_e = min(base + sleep_end_min, horizon), min(base + 1440, horizon)

        if a1_s < a1_e:
            allowed.append((a1_s, a1_e))
        if a2_s < a2_e:
            allowed.append((a2_s, a2_e))
        day += 1

    # cp_model.Domain.FromIntervals needs inclusive bounds: [l, r]
    intervals = []
    for l, r in allowed:
        intervals.append([l, r - 1])
    return cp_model.Domain.FromIntervals(intervals)

def solve_schedule(tasks, durations, n_workers=5, sleep_start="00:30", sleep_end="08:30"):
    # tasks: list[str], durations: list[int] in minutes
    model = cp_model.CpModel()

    def hhmm_to_min(s):
        h, m = map(int, s.split(":"))
        return h * 60 + m

    sleep_s = hhmm_to_min(sleep_start)
    sleep_e = hhmm_to_min(sleep_end)
    assert sleep_s < sleep_e, "這個骨架先假設禁用時段不跨午夜（例如 00:30~08:30）。"

    horizon = sum(durations) + 1440  # buffer 一天
    allowed_domain = build_allowed_start_domain(horizon, sleep_s, sleep_e)

    n = len(tasks)

    # Decision vars
    starts = []
    ends = []
    worker = []
    intervals_by_w = [[] for _ in range(n_workers)]

    for i in range(n):
        s = model.NewIntVarFromDomain(allowed_domain, f"start_{i}")  # start 不可在禁用時段
        e = model.NewIntVar(0, horizon, f"end_{i}")
        model.Add(e == s + durations[i])

        w = model.NewIntVar(0, n_workers - 1, f"worker_{i}")

        # Optional intervals per worker
        opts = []
        for k in range(n_workers):
            is_k = model.NewBoolVar(f"is_{i}_{k}")
            model.Add(w == k).OnlyEnforceIf(is_k)
            model.Add(w != k).OnlyEnforceIf(is_k.Not())
            itv = model.NewOptionalIntervalVar(s, durations[i], e, is_k, f"itv_{i}_{k}")
            intervals_by_w[k].append(itv)
            opts.append(is_k)

        model.AddExactlyOne(opts)

        starts.append(s); ends.append(e); worker.append(w)

    # No overlap on each worker
    for k in range(n_workers):
        model.AddNoOverlap(intervals_by_w[k])

    # Objective (先做最基本：最小化 makespan)
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    out = []
    for i in range(n):
        out.append({
            "task": tasks[i],
            "worker": solver.Value(worker[i]) + 1,
            "start_min": solver.Value(starts[i]),
            "end_min": solver.Value(ends[i]),
        })
    out.sort(key=lambda x: (x["worker"], x["start_min"]))
    return out
