from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st

try:
    from ortools.sat.python import cp_model
except Exception as e:
    cp_model = None
    ORTOOLS_IMPORT_ERROR = e


# =========================
# Domain models
# =========================
@dataclass(frozen=True)
class Building:
    name: str
    duration_min: int


@dataclass(frozen=True)
class WorkerState:
    worker_id: int
    remaining_min: int  # minutes until current upgrade finishes


# =========================
# Time utilities (blocked window)
# =========================
def hm_to_min(hm: str) -> int:
    """'HH:MM' -> minutes since midnight"""
    h, m = hm.strip().split(":")
    return int(h) * 60 + int(m)


def time_to_hm(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"


def is_in_blocked(tod: int, start: int, end: int) -> bool:
    """
    Check if time-of-day (0..1439) is within blocked interval [start, end),
    supporting wrap-around (e.g., 23:30-08:30).
    """
    if start == end:
        return False
    if start < end:
        return start <= tod < end
    return tod >= start or tod < end


def minutes_until_block_end(tod: int, start: int, end: int) -> int:
    """If tod is in blocked window, return minutes to reach the blocked window end."""
    if not is_in_blocked(tod, start, end):
        return 0
    if start < end:
        return end - tod
    # wrap-around
    if tod >= start:
        return (1440 - tod) + end
    return end - tod


def add_minutes(dt: datetime, minutes: int) -> datetime:
    return dt + timedelta(minutes=minutes)


# =========================
# Cost model for "next task only"
# =========================
def compute_worker_earliest_start_offset(
    now: datetime,
    worker: WorkerState,
    blocked_start_min: int,
    blocked_end_min: int,
) -> int:
    """
    Return earliest start time offset (minutes from now) when worker can start next building.
    If current job finishes inside blocked, push to the end of that blocked window.
    """
    now_tod = now.hour * 60 + now.minute
    finish_offset = worker.remaining_min
    finish_tod = (now_tod + finish_offset) % 1440
    wait = minutes_until_block_end(finish_tod, blocked_start_min, blocked_end_min)
    return finish_offset + wait


def compute_idle_after_finish_offset(
    now: datetime,
    start_offset: int,
    duration: int,
    blocked_start_min: int,
    blocked_end_min: int,
) -> int:
    """
    If chosen building finishes inside blocked, worker can't switch -> forced idle until blocked end.
    """
    now_tod = now.hour * 60 + now.minute
    end_offset = start_offset + duration
    end_tod = (now_tod + end_offset) % 1440
    return minutes_until_block_end(end_tod, blocked_start_min, blocked_end_min)


def build_cost_matrix(
    now: datetime,
    workers: List[WorkerState],
    buildings: List[Building],
    blocked_start_min: int,
    blocked_end_min: int,
) -> Tuple[List[List[int]], List[int]]:
    """
    cost[w][b] = forced idle minutes after finishing building b (starting at earliest allowed time for worker w)
    start_offset[w] = earliest start time offset for worker w
    """
    start_offsets = [
        compute_worker_earliest_start_offset(now, w, blocked_start_min, blocked_end_min)
        for w in workers
    ]
    cost = []
    for wi in range(len(workers)):
        row = []
        for b in buildings:
            idle = compute_idle_after_finish_offset(
                now=now,
                start_offset=start_offsets[wi],
                duration=b.duration_min,
                blocked_start_min=blocked_start_min,
                blocked_end_min=blocked_end_min,
            )
            row.append(idle)
        cost.append(row)
    return cost, start_offsets


# =========================
# OR-Tools assignment solver
# =========================
def solve_next_five(
    buildings: List[Building],
    workers: List[WorkerState],
    blocked_start_hm: str,
    blocked_end_hm: str,
    now: Optional[datetime] = None,
    time_limit_sec: float = 3.0,
    fixed_assignments: Optional[dict[int, int]] = None,  # worker_index(0..4) -> building_index(0..B-1)
) -> dict:

    """
    Assign exactly one building to each worker (5 workers => choose 5 buildings),
    minimizing sum of forced idle minutes after completion in blocked window.
    """
    if cp_model is None:
        raise RuntimeError(f"OR-Tools import failed: {ORTOOLS_IMPORT_ERROR}")

    if len(workers) != 5:
        raise ValueError("This app expects exactly 5 workers.")
    if len(buildings) < 5:
        raise ValueError("Need at least 5 buildings to choose from.")

    if now is None:
        now = datetime.now()

    bs = hm_to_min(blocked_start_hm)
    be = hm_to_min(blocked_end_hm)

    cost, start_offsets = build_cost_matrix(now, workers, buildings, bs, be)
    W, B = len(workers), len(buildings)

    model = cp_model.CpModel()
    x = {(w, b): model.NewBoolVar(f"x_w{w}_b{b}") for w in range(W) for b in range(B)}

    if fixed_assignments:
      # Validate: no duplicate building fixed to multiple workers
      fixed_buildings = list(fixed_assignments.values())
      if len(set(fixed_buildings)) != len(fixed_buildings):
          raise ValueError("Invalid fixed_assignments: one building is fixed to multiple workers.")

      for w_idx, b_idx in fixed_assignments.items():
          if not (0 <= w_idx < W):
              raise ValueError(f"Invalid worker index: {w_idx}")
          if not (0 <= b_idx < B):
              raise ValueError(f"Invalid building index: {b_idx}")
          model.Add(x[(w_idx, b_idx)] == 1)


    # each worker gets exactly one building
    for w in range(W):
        model.Add(sum(x[(w, b)] for b in range(B)) == 1)

    # each building used at most once
    for b in range(B):
        model.Add(sum(x[(w, b)] for w in range(W)) <= 1)

    # objective: minimize total forced idle minutes
    model.Minimize(sum(cost[w][b] * x[(w, b)] for w in range(W) for b in range(B)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible assignment found.")

    assignments = []
    total_idle = 0
    for w in range(W):
        chosen_b = None
        for b in range(B):
            if solver.Value(x[(w, b)]) == 1:
                chosen_b = b
                break
        assert chosen_b is not None

        bld = buildings[chosen_b]
        start_offset = start_offsets[w]
        end_offset = start_offset + bld.duration_min
        idle = cost[w][chosen_b]
        total_idle += idle

        assignments.append(
            {
                "Worker": workers[w].worker_id,
                "Current remaining (min)": workers[w].remaining_min,
                "Assigned building": bld.name,
                "Build duration (min)": bld.duration_min,
                "Start time": add_minutes(now, start_offset).strftime("%Y-%m-%d %H:%M"),
                "End time": add_minutes(now, end_offset).strftime("%Y-%m-%d %H:%M"),
                "Forced idle after finish (min)": idle,
            }
        )

    return {
        "now": now.strftime("%Y-%m-%d %H:%M"),
        "blocked_window": f"{blocked_start_hm}~{blocked_end_hm}",
        "total_forced_idle_min": total_idle,
        "assignments": assignments,
    }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CoC Builder Scheduler (Next 5)", layout="wide")
st.title("5 builder optimized assignment.")

with st.expander("Introduction", expanded=True):
    st.markdown(
        """
- Enter building name and upgrade time
- Input the current "remaining completion time" for 5 builders
- Set the blocked window (default 00:30-08:30, adjustable)
- Click the button: It will select the next 5 buildings and assign them to the 5 bulders, aiming to **minimize forced idle minutes caused by completion during blocked hours**
        """.strip()
    )

if cp_model is None:
    st.error(f"Error:{ORTOOLS_IMPORT_ERROR}")
    st.stop()

colA, colB = st.columns([1.1, 0.9])

import pandas as pd
import streamlit as st

with colA:
    st.subheader("1) Building list (select rows → delete or assign worker)")

    if "bld_df" not in st.session_state:
        st.session_state.bld_df = pd.DataFrame(
            [
                {"name": "Archer Tower", "duration_hours": 6, "duration_minutes": 0},
                {"name": "Air Defense", "duration_hours": 14, "duration_minutes": 0},
                {"name": "Gold Storage", "duration_hours": 8, "duration_minutes": 0},
                {"name": "Barracks", "duration_hours": 3, "duration_minutes": 20},
                {"name": "Wizard Tower", "duration_hours": 10, "duration_minutes": 0},
            ]
        )

    if "worker_fixed_rowid" not in st.session_state:
        # worker_id (1..5) -> rowid in bld_df, or None
        st.session_state.worker_fixed_rowid = {i: None for i in range(1, 6)}

    # Display table for selection (not editing)
    display_df = st.session_state.bld_df.reset_index().rename(columns={"index": "_rowid"})

    sel = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="bld_select_table",
    )

    # Read selected rows (compatible across some Streamlit versions)
    selected_rows = []
    try:
        selected_rows = list(sel.selection.rows)
    except Exception:
        try:
            selected_rows = list(sel["selection"]["rows"])
        except Exception:
            selected_rows = []

    chosen_rowids = display_df.loc[selected_rows, "_rowid"].tolist() if selected_rows else []
    st.caption(f"Selected rows: {len(chosen_rowids)}")

    action = st.radio(
        "Action for selected rows",
        ["None", "Delete selected rows", "Assign to a worker (fixed)"],
        horizontal=True,
        key="bld_action",
    )

    if action == "Assign to a worker (fixed)":
        left, right = st.columns([1, 2])
        with left:
            target_worker = st.selectbox("Worker", [1, 2, 3, 4, 5], index=0, key="target_worker")
        with right:
            st.info("Select exactly one building row, then apply the assignment.")

        if st.button("Apply fixed assignment", use_container_width=True, key="apply_fixed"):
            if len(chosen_rowids) != 1:
                st.error("For assignment, please select exactly one row.")
            else:
                rowid = int(chosen_rowids[0])
                st.session_state.worker_fixed_rowid[target_worker] = rowid
                st.success(f"Fixed: Worker {target_worker} -> row {rowid}")
            st.rerun()

        # Show current fixed assignments
        fixed_view = []
        for w in range(1, 6):
            rid = st.session_state.worker_fixed_rowid.get(w)
            if rid is None:
                fixed_view.append({"Worker": w, "Fixed building": "(none)"})
            else:
                b = st.session_state.bld_df.iloc[rid]
                fixed_view.append(
                    {
                        "Worker": w,
                        "Fixed building": f"{b['name']} ({int(b['duration_hours'])}h {int(b['duration_minutes'])}m) [row={rid}]",
                    }
                )
        st.dataframe(pd.DataFrame(fixed_view), use_container_width=True, hide_index=True)

        if st.button("Clear all fixed assignments", use_container_width=True, key="clear_fixed"):
            st.session_state.worker_fixed_rowid = {i: None for i in range(1, 6)}
            st.rerun()

    if action == "Delete selected rows":
        if st.button("Delete now", type="primary", use_container_width=True, key="delete_rows"):
            if not chosen_rowids:
                st.warning("No rows selected.")
            else:
                # Clear fixed assignments that point to deleted rows
                deleted_set = set(int(x) for x in chosen_rowids)
                for w in range(1, 6):
                    rid = st.session_state.worker_fixed_rowid.get(w)
                    if rid in deleted_set:
                        st.session_state.worker_fixed_rowid[w] = None

                st.session_state.bld_df = (
                    st.session_state.bld_df.drop(index=[int(x) for x in chosen_rowids]).reset_index(drop=True)
                )
                st.success(f"Deleted rows: {len(chosen_rowids)}")
            st.rerun()

    # Optional: editing panel (separate from selection table)
    tab_select, tab_edit = st.tabs(["Select", "Edit"])

    with tab_edit:
        st.caption("Edit here, then click Apply. This avoids rerun lag while typing.")

        if "bld_df_draft" not in st.session_state:
            st.session_state.bld_df_draft = st.session_state.bld_df.copy()

        draft = st.data_editor(
            st.session_state.bld_df_draft,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Building name"),
                "duration_hours": st.column_config.NumberColumn("Hours", min_value=0, step=1),
                "duration_minutes": st.column_config.NumberColumn("Minutes", min_value=0, max_value=59, step=5),
            },
            key="bld_editor_draft",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Apply changes", type="primary", use_container_width=True):
                st.session_state.bld_df = draft.copy()
                st.session_state.bld_df_draft = st.session_state.bld_df.copy()
                st.toast("Saved.")
                st.rerun()

        with c2:
            if st.button("Discard changes", use_container_width=True):
                st.session_state.bld_df_draft = st.session_state.bld_df.copy()
                st.toast("Discarded.")
                st.rerun()


with colB:
    st.subheader("2) remaining completion time")
    st.caption("Fill 0 if builders are available。")

    worker_rows = []
    for i in range(1, 6):
        c1, c2 = st.columns(2)
        with c1:
            rh = st.number_input(f"Worker {i} remaining hours", min_value=0, value=0, step=1, key=f"w{i}_h")
        with c2:
            rm = st.number_input(f"Worker {i} remaining minutes", min_value=0, max_value=59, value=0, step=5, key=f"w{i}_m")
        worker_rows.append((i, int(rh), int(rm)))

    st.subheader("3) Set the blocked window")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        blocked_start_t = st.time_input("Blocked start", value=time(0, 30))
    with c2:
        blocked_end_t = st.time_input("Blocked end", value=time(8, 30))
    with c3:
        time_limit = st.number_input("OR-Tools time limit (sec)", min_value=0.1, value=3.0, step=0.5)

    st.subheader("4) Current time")
    use_custom_now = st.toggle("Manually specify the time (Default is current time)", value=False)
    if use_custom_now:
        now_date = st.date_input("Date")
        now_time = st.time_input("Time", value=datetime.now().time().replace(second=0, microsecond=0))
        now = datetime.combine(now_date, now_time)
    else:
        now = datetime.now()


def parse_buildings(df: pd.DataFrame) -> List[Building]:
    buildings: List[Building] = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        h = int(row.get("duration_hours", 0) or 0)
        m = int(row.get("duration_minutes", 0) or 0)
        if h < 0 or m < 0 or m >= 60:
            continue
        dur = h * 60 + m
        if dur <= 0:
            continue
        buildings.append(Building(name=name, duration_min=dur))
    return buildings


def parse_workers(rows: List[Tuple[int, int, int]]) -> List[WorkerState]:
    workers: List[WorkerState] = []
    for wid, h, m in rows:
        workers.append(WorkerState(worker_id=wid, remaining_min=int(h) * 60 + int(m)))
    return workers


st.divider()
run = st.button("✅ Find optimal next 5 buildings assignment", type="primary", use_container_width=True)

if run:
    buildings = parse_buildings(st.session_state.bld_df)
    workers = parse_workers(worker_rows)

    blocked_start_hm = time_to_hm(blocked_start_t)
    blocked_end_hm = time_to_hm(blocked_end_t)

    if len(buildings) < 5:
        st.error("At least 5 items are necessary。")
        st.stop()

    # de-duplicate building names if user repeats them (keep as-is but warn)
    names = [b.name for b in buildings]
    if len(set(names)) != len(names):
        st.warning("Detected duplicate building names.")

    try:
        fixed_assignments = {}
        for worker_id, rowid in st.session_state.worker_fixed_rowid.items():
            if rowid is None:
                continue
            # worker_index is 0..4
            w_idx = worker_id - 1
            b_idx = int(rowid)
            fixed_assignments[w_idx] = b_idx
        result = solve_next_five(
            buildings=buildings,
            workers=workers,
            blocked_start_hm=blocked_start_hm,
            blocked_end_hm=blocked_end_hm,
            now=now,
            time_limit_sec=float(time_limit),
            fixed_assignments=fixed_assignments,
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    st.success(
        f"Finish! Blocked time={result['blocked_window']}，"
        f"total forced idle time={result['total_forced_idle_min']} mins"
    )

    out_df = pd.DataFrame(result["assignments"]).sort_values("Worker")
    st.dataframe(out_df, use_container_width=True)

    # Optional: show summary
    st.subheader("Summary")
    st.write(
        f"Now: {result['now']}  |  Blocked window: {result['blocked_window']}  |  Total forced idle: {result['total_forced_idle_min']} min"
    )

    csv = out_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download result CSV", data=csv, file_name="coc_next5_assignment.csv", mime="text/csv")
