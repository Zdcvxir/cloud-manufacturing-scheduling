"""Variable Neighborhood Search (VNS) for the scheduling problem.

This file implements the VNS used in the computational experiments. Job and
slot indices are one-based inside schedules to stay close to the notation in
the paper; Python list indices are converted at the boundary.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Callable

import config


N = config.N
T = config.T
B = config.B
a = config.a
seta = config.seta
w = config.w
u = config.u
k = config.k
lam = config.lam
c = config.c
tao_l = config.tao_l
p_j = config.p_j
L_l = config.L_l
CP_l = config.CP_l

INFEASIBLE_COST = 1e6
BLOCK_RATIO = 0.1
BLOCK_LEN = max(1, math.ceil(BLOCK_RATIO * N))

Schedule = dict[int, list[int]]
Assignment = list[int]
History = list[tuple[float, float]]
Neighborhood = Callable[[Assignment, float, float], Assignment]


def print_instance_summary() -> None:
    """Print the key instance parameters used by this run."""
    print(
        f"Instance: N={N}, T={T}, B={B}, c={c}, "
        f"a={a}, setup={seta}, w={w}, u={u}"
    )


def sorted_jobs_by_processing_time() -> list[tuple[int, int]]:
    """Return jobs in non-increasing processing-time order."""
    return sorted(((job_id + 1, p_j[job_id]) for job_id in range(N)), key=lambda item: item[1], reverse=True)


def build_initial_solution() -> Schedule:
    """Construct an initial feasible solution using the greedy rule in Algorithm 3."""
    slot_jobs: Schedule = {slot: [] for slot in range(1, T + 1)}
    slot_used_time = {slot: 0.0 for slot in range(1, T + 1)}

    global_job_position = 1
    current_slot = 1

    for job_id, processing_time in sorted_jobs_by_processing_time():
        allocated = False

        for slot_id in range(current_slot, T + 1):
            current_job_count = len(slot_jobs[slot_id])
            new_job_count = current_job_count + 1
            actual_processing = processing_time * (global_job_position**a)
            batches_before = math.ceil(current_job_count / c) if current_job_count > 0 else 0
            batches_after = math.ceil(new_job_count / c)
            extra_setup = (batches_after - batches_before) * seta
            total_after = slot_used_time[slot_id] + actual_processing + extra_setup

            if total_after <= L_l[slot_id - 1]:
                slot_jobs[slot_id].append(job_id)
                slot_used_time[slot_id] = total_after
                global_job_position += 1
                current_slot = max(current_slot, slot_id)
                allocated = True
                break

        if not allocated:
            raise RuntimeError(
                f"Initial solution failed: job {job_id} cannot be placed in slots "
                f"{current_slot}..{T}."
            )

    return {slot: jobs for slot, jobs in slot_jobs.items() if jobs}


def evaluate_schedule(schedule: Schedule) -> tuple[float, list[int]]:
    """Evaluate a slot assignment and compute the best maintenance plan by DP."""
    selected_slots = sorted(schedule)
    num_selected = len(selected_slots)
    if num_selected == 0:
        return 0.0, []

    job_counts: list[int] = []
    processing_times: list[list[int]] = []
    tau_values: list[float] = []
    fixed_costs: list[float] = []

    for slot_id in selected_slots:
        job_ids = schedule[slot_id]
        job_counts.append(len(job_ids))
        processing_times.append([p_j[job_id - 1] for job_id in job_ids])
        tau_values.append(tao_l[slot_id - 1])
        fixed_costs.append(CP_l[slot_id - 1])

    slot_processing_times = [0.0] * num_selected
    jobs_before_slot = 0

    for slot_index, job_count in enumerate(job_counts):
        num_batches = math.ceil(job_count / c)
        setup_time = num_batches * seta
        processing_time = 0.0

        for job_index in range(job_count):
            global_position = jobs_before_slot + job_index + 1
            processing_time += processing_times[slot_index][job_index] * (global_position**a)

        slot_processing_times[slot_index] = setup_time + processing_time

        if slot_processing_times[slot_index] > L_l[selected_slots[slot_index] - 1]:
            return INFEASIBLE_COST, []

        jobs_before_slot += job_count

    prefix_processing = [0.0] * (num_selected + 1)
    prefix_service = [0.0] * (num_selected + 1)
    prefix_fixed = [0.0] * (num_selected + 1)
    prefix_cumulative_processing = [0.0] * (num_selected + 1)

    for end in range(1, num_selected + 1):
        index = end - 1
        prefix_processing[end] = prefix_processing[end - 1] + slot_processing_times[index]
        prefix_service[end] = prefix_service[end - 1] + tau_values[index] * slot_processing_times[index]
        prefix_fixed[end] = prefix_fixed[end - 1] + fixed_costs[index]
        prefix_cumulative_processing[end] = prefix_cumulative_processing[end - 1] + prefix_processing[end]

    repair_factor = w * k * lam
    best_cost = [float("inf")] * (num_selected + 1)
    best_cost[0] = 0.0
    predecessor = [-1] * (num_selected + 1)

    for end in range(1, num_selected + 1):
        for start in range(end - 1, -1, -1):
            service_cost = prefix_service[end] - prefix_service[start]
            fixed_cost = prefix_fixed[end] - prefix_fixed[start]
            repair_cost = repair_factor * (
                (prefix_cumulative_processing[end] - prefix_cumulative_processing[start])
                - (end - start) * prefix_processing[start]
            )
            candidate = best_cost[start] + u + service_cost + fixed_cost + repair_cost

            if candidate < best_cost[end]:
                best_cost[end] = candidate
                predecessor[end] = start

    breakpoints: list[int] = []
    end = num_selected
    while predecessor[end] != -1:
        breakpoints.append(predecessor[end])
        end = predecessor[end]
    breakpoints.reverse()

    maintenance_slots = [selected_slots[index] for index in breakpoints[1:]]
    objective_value = best_cost[num_selected] - u
    return objective_value, maintenance_slots


def encode_solution(slot_jobs: Schedule) -> Assignment:
    """Convert a slot-to-jobs dictionary into a job-to-slot assignment list."""
    assignment = [0] * N
    for slot_id, jobs in slot_jobs.items():
        for job_id in jobs:
            assignment[job_id - 1] = slot_id
    return assignment


def decode_solution(assignment: Assignment) -> Schedule:
    """Convert a job-to-slot assignment list into sorted slot job lists."""
    slot_jobs: Schedule = {slot: [] for slot in range(1, T + 1)}
    for job_index, slot_id in enumerate(assignment):
        if slot_id != 0:
            slot_jobs[slot_id].append(job_index + 1)

    for jobs in slot_jobs.values():
        jobs.sort(key=lambda job_id: p_j[job_id - 1], reverse=True)

    return {slot: jobs for slot, jobs in slot_jobs.items() if jobs}


def shake_n1(assignment: Assignment, start_time: float, time_limit: float) -> Assignment:
    """Randomly reassign two jobs."""
    del start_time, time_limit
    new_assignment = assignment[:]
    job_1, job_2 = random.sample(range(N), 2)
    new_assignment[job_1] = random.randint(1, T)
    new_assignment[job_2] = random.randint(1, T)
    return new_assignment


def shake_n2(assignment: Assignment, start_time: float, time_limit: float) -> Assignment:
    """Swap two blocks of assignments."""
    del start_time, time_limit
    if N < 2 * BLOCK_LEN:
        return _swap_two_jobs(assignment)

    for _ in range(100):
        start_1 = random.randint(0, N - BLOCK_LEN)
        start_2 = random.randint(0, N - BLOCK_LEN)
        if abs(start_1 - start_2) >= BLOCK_LEN:
            new_assignment = assignment[:]
            block_1 = new_assignment[start_1 : start_1 + BLOCK_LEN]
            block_2 = new_assignment[start_2 : start_2 + BLOCK_LEN]
            new_assignment[start_1 : start_1 + BLOCK_LEN] = block_2
            new_assignment[start_2 : start_2 + BLOCK_LEN] = block_1
            return new_assignment

    return _swap_two_jobs(assignment)


def shake_n3(assignment: Assignment, start_time: float, time_limit: float) -> Assignment:
    """Remove one occupied slot and greedily reinsert its jobs."""
    slot_jobs = _assignment_to_full_slot_dict(assignment)
    non_empty_slots = [slot for slot, jobs in slot_jobs.items() if jobs]
    if not non_empty_slots:
        return assignment[:]

    removed_slot = random.choice(non_empty_slots)
    removed_jobs = slot_jobs[removed_slot][:]
    random.shuffle(removed_jobs)
    slot_jobs[removed_slot] = []

    for job_id in removed_jobs:
        if _time_is_up(start_time, time_limit):
            return assignment[:]

        best_cost = float("inf")
        best_slot = None
        best_slot_jobs: list[int] | None = None

        for slot_id in range(1, T + 1):
            if _time_is_up(start_time, time_limit):
                return assignment[:]

            trial_slots = _copy_full_slot_dict(slot_jobs)
            trial_slots[slot_id] = trial_slots[slot_id] + [job_id]
            trial_slots[slot_id].sort(key=lambda candidate: p_j[candidate - 1], reverse=True)
            trial_cost = _slot_dict_to_cost(trial_slots)

            if trial_cost < best_cost:
                best_cost = trial_cost
                best_slot = slot_id
                best_slot_jobs = trial_slots[slot_id]

        if best_slot is None or best_slot_jobs is None:
            slot_jobs[removed_slot].append(job_id)
            slot_jobs[removed_slot].sort(key=lambda candidate: p_j[candidate - 1], reverse=True)
        else:
            slot_jobs[best_slot] = best_slot_jobs

    return _full_slot_dict_to_assignment(slot_jobs)


def shake_n4(assignment: Assignment, start_time: float, time_limit: float) -> Assignment:
    """Remove two occupied slots and reinsert jobs by regret ordering."""
    slot_jobs = _assignment_to_full_slot_dict(assignment)
    non_empty_slots = [slot for slot, jobs in slot_jobs.items() if jobs]
    if len(non_empty_slots) < 2:
        return assignment[:]

    slot_1, slot_2 = random.sample(non_empty_slots, 2)
    removed_jobs = slot_jobs[slot_1] + slot_jobs[slot_2]
    slot_jobs[slot_1] = []
    slot_jobs[slot_2] = []

    regret_info: list[tuple[int, float]] = []
    for job_id in removed_jobs:
        if _time_is_up(start_time, time_limit):
            return assignment[:]

        best_cost = float("inf")
        second_best_cost = float("inf")

        for slot_id in range(1, T + 1):
            if _time_is_up(start_time, time_limit):
                return assignment[:]

            trial_slots = _copy_full_slot_dict(slot_jobs)
            trial_slots[slot_id] = trial_slots[slot_id] + [job_id]
            trial_slots[slot_id].sort(key=lambda candidate: p_j[candidate - 1], reverse=True)
            trial_cost = _slot_dict_to_cost(trial_slots)

            if trial_cost < best_cost:
                second_best_cost = best_cost
                best_cost = trial_cost
            elif trial_cost < second_best_cost:
                second_best_cost = trial_cost

        regret = second_best_cost - best_cost if second_best_cost != float("inf") else 0.0
        regret_info.append((job_id, regret))

    regret_info.sort(key=lambda item: item[1], reverse=True)

    for job_id, _ in regret_info:
        if _time_is_up(start_time, time_limit):
            return assignment[:]

        best_cost = float("inf")
        best_slot = None
        best_slot_jobs: list[int] | None = None

        for slot_id in range(1, T + 1):
            if _time_is_up(start_time, time_limit):
                return assignment[:]

            new_jobs = slot_jobs[slot_id][:] + [job_id]
            new_jobs.sort(key=lambda candidate: p_j[candidate - 1], reverse=True)
            trial_slots = _copy_full_slot_dict(slot_jobs)
            trial_slots[slot_id] = new_jobs
            trial_cost = _slot_dict_to_cost(trial_slots)

            if trial_cost < best_cost:
                best_cost = trial_cost
                best_slot = slot_id
                best_slot_jobs = new_jobs

        if best_slot is None or best_slot_jobs is None:
            fallback_slot = random.randint(1, T)
            slot_jobs[fallback_slot].append(job_id)
            slot_jobs[fallback_slot].sort(key=lambda candidate: p_j[candidate - 1], reverse=True)
        else:
            slot_jobs[best_slot] = best_slot_jobs

    return _full_slot_dict_to_assignment(slot_jobs)


NEIGHBORHOODS: list[Neighborhood] = [shake_n1, shake_n2, shake_n3, shake_n4]


def local_search(assignment: Assignment, start_time: float, time_limit: float) -> Assignment:
    """Improve an assignment with the three local search neighborhoods."""
    current_assignment = assignment[:]
    current_cost, _ = evaluate_schedule(decode_solution(current_assignment))

    def apply_l1() -> bool:
        nonlocal current_assignment, current_cost
        job_id = random.randint(0, N - 1)
        old_slot = current_assignment[job_id]
        candidate_slots = [slot for slot in range(1, T + 1) if slot != old_slot]
        random.shuffle(candidate_slots)

        for new_slot in candidate_slots:
            if _time_is_up(start_time, time_limit):
                return False

            new_assignment = current_assignment[:]
            new_assignment[job_id] = new_slot
            new_cost, _ = evaluate_schedule(decode_solution(new_assignment))

            if new_cost < current_cost:
                current_assignment = new_assignment
                current_cost = new_cost
                return True

        return False

    def apply_l2() -> bool:
        nonlocal current_assignment, current_cost
        if _time_is_up(start_time, time_limit):
            return False

        job_1 = random.randint(0, N - 1)
        slot_1 = current_assignment[job_1]
        occupied_slots = set(current_assignment)
        candidate_slots = [slot for slot in occupied_slots if slot not in {0, slot_1}]
        if not candidate_slots:
            return False

        slot_2 = random.choice(candidate_slots)
        job_2_candidates = [job for job, slot in enumerate(current_assignment) if slot == slot_2]
        random.shuffle(job_2_candidates)

        for job_2 in job_2_candidates:
            if _time_is_up(start_time, time_limit):
                return False

            new_assignment = current_assignment[:]
            new_assignment[job_1], new_assignment[job_2] = new_assignment[job_2], new_assignment[job_1]
            new_cost, _ = evaluate_schedule(decode_solution(new_assignment))

            if new_cost < current_cost:
                current_assignment = new_assignment
                current_cost = new_cost
                return True

        return False

    def apply_l3() -> bool:
        nonlocal current_assignment, current_cost
        if _time_is_up(start_time, time_limit):
            return False

        slot_to_jobs: dict[int, list[int]] = {}
        for job, slot in enumerate(current_assignment):
            if slot != 0:
                slot_to_jobs.setdefault(slot, []).append(job)

        source_slots = [slot for slot, jobs in slot_to_jobs.items() if len(jobs) >= 2]
        if not source_slots:
            return False

        source_slot = random.choice(source_slots)
        job_1, job_2 = random.sample(slot_to_jobs[source_slot], 2)

        target_slots = [slot for slot in slot_to_jobs if slot != source_slot]
        if not target_slots:
            return False

        target_slot = random.choice(target_slots)
        target_jobs = slot_to_jobs[target_slot][:]
        random.shuffle(target_jobs)

        for job_3 in target_jobs:
            if _time_is_up(start_time, time_limit):
                return False

            new_assignment = current_assignment[:]
            new_assignment[job_1] = target_slot
            new_assignment[job_2] = target_slot
            new_assignment[job_3] = source_slot
            new_cost, _ = evaluate_schedule(decode_solution(new_assignment))

            if new_cost < current_cost:
                current_assignment = new_assignment
                current_cost = new_cost
                return True

        return False

    local_neighborhoods = [apply_l1, apply_l2, apply_l3]

    while not _time_is_up(start_time, time_limit):
        random.shuffle(local_neighborhoods)
        improved = False

        for neighborhood in local_neighborhoods:
            if _time_is_up(start_time, time_limit):
                break
            if neighborhood():
                improved = True
                break

        if not improved:
            break

    return current_assignment


def variable_neighborhood_search(
    initial_assignment: Assignment,
    time_limit: float,
    verbose: bool = True,
) -> tuple[Assignment, float, list[int], History]:
    """Run the VNS metaheuristic from an initial assignment."""
    best_assignment = initial_assignment[:]
    best_cost, best_pm_slots = evaluate_schedule(decode_solution(best_assignment))
    start_time = time.time()
    history: History = [(0.0, best_cost)]
    improvement_count = 0

    if verbose:
        print("\n--- VNS Improvements ---")
        print(f"{'Time':>8s}  {'Impr#':>6s}  {'Cost':>12s}  {'Neigh':>6s}")
        print("-" * 42)

    while not _time_is_up(start_time, time_limit):
        neighborhood_index = 0

        while neighborhood_index < len(NEIGHBORHOODS):
            if _time_is_up(start_time, time_limit):
                break

            candidate_assignment = NEIGHBORHOODS[neighborhood_index](
                best_assignment,
                start_time,
                time_limit,
            )
            candidate_assignment = local_search(candidate_assignment, start_time, time_limit)
            candidate_cost, candidate_pm_slots = evaluate_schedule(decode_solution(candidate_assignment))

            if candidate_cost < best_cost:
                improvement_count += 1
                best_cost = candidate_cost
                best_assignment = candidate_assignment[:]
                best_pm_slots = candidate_pm_slots
                elapsed = time.time() - start_time
                history.append((elapsed, best_cost))

                if verbose:
                    print(
                        f"{elapsed:8.2f}  {improvement_count:6d}  "
                        f"{best_cost:12.2f}  {neighborhood_index + 1:6d}"
                    )

                neighborhood_index = 0
            else:
                neighborhood_index += 1

    total_elapsed = time.time() - start_time
    if history[-1][0] < total_elapsed:
        history.append((total_elapsed, best_cost))

    if verbose:
        print("-" * 42)

    return best_assignment, best_cost, best_pm_slots, history


def interpolate_cost(history: History, target_time: float) -> float:
    """Return the best-known cost at target_time using a step function."""
    if target_time <= history[0][0]:
        return history[0][1]
    if target_time >= history[-1][0]:
        return history[-1][1]

    for index in range(len(history) - 1):
        if history[index][0] <= target_time < history[index + 1][0]:
            return history[index][1]

    return history[-1][1]


def run_experiments(args: argparse.Namespace) -> None:
    """Run one or more independent VNS experiments."""
    config.validate_instance()
    time_limit = args.time_limit if args.time_limit is not None else default_time_limit()
    output_path = args.output or Path(__file__).with_name(f"{N}-{T}.txt")

    print_instance_summary()

    if args.runs == 1:
        random.seed(args.seed)
        print("Generating initial solution...")
        initial_schedule = build_initial_solution()
        initial_assignment = encode_solution(initial_schedule)
        initial_cost, _ = evaluate_schedule(initial_schedule)

        print("\n===== Initial Solution =====")
        print(f"Total cost: {initial_cost:.2f}")

        print("\n===== Running VNS =====")
        best_assignment, best_cost, _, history = variable_neighborhood_search(
            initial_assignment,
            time_limit,
            verbose=not args.quiet,
        )
        best_schedule = decode_solution(best_assignment)
        print_single_run_summary(best_schedule, best_cost)

        if not args.no_output:
            write_single_run_curve(output_path, history, time_limit, best_cost)
            print(f"Convergence curve saved to {output_path}")

        return

    print(f"\n===== Running {args.runs} independent experiments ({time_limit:.2f}s each) =====")
    costs: list[float] = []
    histories: list[History] = []
    occupancies: list[int] = []

    for run in range(1, args.runs + 1):
        random.seed(args.seed + run)
        print(f"\nRun {run}/{args.runs}...")
        initial_assignment = encode_solution(build_initial_solution())
        best_assignment, best_cost, _, history = variable_neighborhood_search(
            initial_assignment,
            time_limit,
            verbose=False,
        )
        best_schedule = decode_solution(best_assignment)

        costs.append(best_cost)
        histories.append(history)
        occupancies.append(len(best_schedule))
        print(f"Run {run}: best cost = {best_cost:.2f}")

    print_multi_run_summary(costs, occupancies)

    if not args.no_output:
        write_average_curve(output_path, histories, time_limit, costs)
        print(f"Average convergence curve saved to {output_path}")


def print_single_run_summary(
    best_schedule: Schedule,
    best_cost: float,
) -> None:
    """Print solution details for a single run."""
    print(f"\nBest total cost: {best_cost:.2f}")
    print("\n===== Best Schedule =====")
    for slot in sorted(best_schedule):
        print(f"Slot {slot}: jobs {best_schedule[slot]}")

    occupied_slots = len(best_schedule)
    print(f"Slot occupancy: {occupied_slots}/{T} = {occupied_slots / T * 100:.1f}%")


def print_multi_run_summary(
    costs: list[float],
    occupancies: list[int],
) -> None:
    """Print aggregate statistics across independent runs."""
    num_runs = len(costs)
    average_cost = sum(costs) / num_runs
    average_occupancy = sum(occupancies) / num_runs

    print("\n===== Statistical Results =====")
    print(f"Number of runs: {num_runs}")
    print(f"Best cost:      {min(costs):.2f}")
    print(f"Average cost:   {average_cost:.2f}")
    print(f"Worst cost:     {max(costs):.2f}")
    print(f"Average slot occupancy: {average_occupancy:.1f}/{T} = {average_occupancy / T * 100:.1f}%")


def write_single_run_curve(
    output_path: Path,
    history: History,
    time_limit: float,
    best_cost: float,
) -> None:
    """Write the convergence curve for one run."""
    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"# N={N}  T={T}  Runs=1  TimeLimit={time_limit}s  BestCost={best_cost:.2f}\n")
        file.write("Time\tCost\n")
        for elapsed, cost in history:
            file.write(f"{elapsed:.3f}\t{cost:.2f}\n")


def write_average_curve(
    output_path: Path,
    histories: list[History],
    time_limit: float,
    costs: list[float],
    step: float = 0.1,
) -> None:
    """Write the average convergence curve across multiple runs."""
    sample_times = [index * step for index in range(int(time_limit / step) + 1)]
    if not sample_times or sample_times[-1] < time_limit:
        sample_times.append(time_limit)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"# N={N}  T={T}  Runs={len(histories)}  TimeLimit={time_limit}s\n")
        file.write(
            f"# AvgCost={sum(costs) / len(costs):.2f}  "
            f"BestCost={min(costs):.2f}  WorstCost={max(costs):.2f}\n"
        )
        file.write("Time\tCost\t" + "\t".join(f"Run{index}" for index in range(1, len(histories) + 1)) + "\n")

        for sample_time in sample_times:
            run_values = [interpolate_cost(history, sample_time) for history in histories]
            mean_value = sum(run_values) / len(run_values)
            file.write(
                f"{sample_time:.1f}\t{mean_value:.2f}\t"
                + "\t".join(f"{value:.2f}" for value in run_values)
                + "\n"
            )


def default_time_limit() -> float:
    """Use the same default time limit as the original experimental scripts."""
    return 20.0 if N >= 100 else 10.0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="number of independent runs")
    parser.add_argument("--time-limit", type=float, default=None, help="time limit per run in seconds")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--output", type=Path, default=None, help="path for convergence-curve output")
    parser.add_argument("--no-output", action="store_true", help="do not write convergence-curve files")
    parser.add_argument("--quiet", action="store_true", help="suppress per-improvement logs for a single run")
    return parser


def _swap_two_jobs(assignment: Assignment) -> Assignment:
    new_assignment = assignment[:]
    job_1, job_2 = random.sample(range(N), 2)
    new_assignment[job_1], new_assignment[job_2] = new_assignment[job_2], new_assignment[job_1]
    return new_assignment


def _assignment_to_full_slot_dict(assignment: Assignment) -> dict[int, list[int]]:
    slot_jobs = {slot: [] for slot in range(1, T + 1)}
    for job_index, slot_id in enumerate(assignment):
        if slot_id != 0:
            slot_jobs[slot_id].append(job_index + 1)

    for jobs in slot_jobs.values():
        jobs.sort(key=lambda job_id: p_j[job_id - 1], reverse=True)

    return slot_jobs


def _copy_full_slot_dict(slot_jobs: dict[int, list[int]]) -> dict[int, list[int]]:
    return {slot: jobs[:] for slot, jobs in slot_jobs.items()}


def _full_slot_dict_to_assignment(slot_jobs: dict[int, list[int]]) -> Assignment:
    assignment = [0] * N
    for slot_id, jobs in slot_jobs.items():
        for job_id in jobs:
            assignment[job_id - 1] = slot_id
    return assignment


def _slot_dict_to_cost(slot_jobs: dict[int, list[int]]) -> float:
    non_empty_slots = {slot: jobs for slot, jobs in slot_jobs.items() if jobs}
    cost, _ = evaluate_schedule(non_empty_slots)
    return cost


def _time_is_up(start_time: float, time_limit: float) -> bool:
    return time.time() - start_time >= time_limit


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be a positive integer.")
    if args.time_limit is not None and args.time_limit < 0:
        raise SystemExit("--time-limit must be non-negative.")
    run_experiments(args)


if __name__ == "__main__":
    main()
