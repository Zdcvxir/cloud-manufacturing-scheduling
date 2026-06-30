"""Less-is-more Variable Neighborhood Search (LIM-VNS).

This file implements the LIM-VNS used for comparative evaluation. It shares
the dynamic-programming cost evaluation in ``vns.py`` and uses a deliberately
streamlined search design.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import config
from vns import (
    N,
    T,
    Assignment,
    History,
    INFEASIBLE_COST,
    Schedule,
    decode_solution,
    default_time_limit,
    encode_solution,
    evaluate_schedule,
    interpolate_cost,
    print_instance_summary,
    print_multi_run_summary,
    print_single_run_summary,
    write_average_curve,
    write_single_run_curve,
)


SHAKE_MAX = 5
LOCAL_SEARCH_MAX_TRIALS = 50
MAX_NO_IMPROVE = 10
MAX_STAGNATION = 60


def random_initial_solution(max_attempts: int = 1000) -> Schedule:
    """Generate the best feasible random initial solution found within a budget."""
    best_cost = float("inf")
    best_schedule: Schedule | None = None

    for _ in range(max_attempts):
        assignment = [random.randint(1, T) for _ in range(N)]
        schedule = decode_solution(assignment)
        cost, _ = evaluate_schedule(schedule)

        if cost < INFEASIBLE_COST and cost < best_cost:
            best_cost = cost
            best_schedule = schedule

    if best_schedule is None:
        raise RuntimeError(
            "No feasible random initial solution was found after "
            f"{max_attempts} attempts."
        )

    return best_schedule


def shake_insert(assignment: Assignment, strength: int = 1) -> Assignment:
    """Move ``strength`` randomly selected jobs to different random slots."""
    new_assignment = assignment[:]

    for _ in range(strength):
        job = random.randint(0, N - 1)
        old_slot = new_assignment[job]
        candidate_slots = [slot for slot in range(1, T + 1) if slot != old_slot]

        if candidate_slots:
            new_assignment[job] = random.choice(candidate_slots)

    return new_assignment


def local_search(assignment: Assignment, start_time: float, time_limit: float) -> Assignment:
    """Run a compact random local search using swap and insertion moves."""
    current_assignment = assignment[:]
    current_cost, _ = evaluate_schedule(decode_solution(current_assignment))
    no_improve_batches = 0

    while no_improve_batches < MAX_NO_IMPROVE:
        improved = False

        for _ in range(LOCAL_SEARCH_MAX_TRIALS):
            if _time_is_up(start_time, time_limit):
                return current_assignment

            if random.random() < 0.5 and len(set(current_assignment)) > 1:
                new_assignment = _random_swap(current_assignment)
                if new_assignment is None:
                    continue
            else:
                new_assignment = _random_insert(current_assignment)
                if new_assignment is None:
                    continue

            new_cost, _ = evaluate_schedule(decode_solution(new_assignment))

            if new_cost < current_cost:
                current_assignment = new_assignment
                current_cost = new_cost
                improved = True
                break

        no_improve_batches = 0 if improved else no_improve_batches + 1

    return current_assignment


def generate_random_feasible_solution(
    start_time: float,
    time_limit: float,
    max_trials: int = 1000,
) -> Assignment | None:
    """Generate a feasible solution for restart."""
    for _ in range(max_trials):
        if _time_is_up(start_time, time_limit):
            return None

        random_assignment = [random.randint(1, T) for _ in range(N)]
        cost, _ = evaluate_schedule(decode_solution(random_assignment))

        if cost < INFEASIBLE_COST:
            return local_search(random_assignment, start_time, time_limit)

    return None


def simplified_vns(
    initial_assignment: Assignment,
    time_limit: float,
    verbose: bool = True,
) -> tuple[Assignment, float, list[int], History]:
    """Run the LIM-VNS metaheuristic."""
    current_assignment = initial_assignment[:]
    current_cost, current_pm_slots = evaluate_schedule(decode_solution(current_assignment))

    best_assignment = current_assignment[:]
    best_cost = current_cost
    best_pm_slots = current_pm_slots

    start_time = time.time()
    shake_strength = 1
    stagnation = 0
    improvement_count = 0
    restart_triggered = False
    history: History = [(0.0, best_cost)]

    if verbose:
        print("\n--- LIM-VNS Progress ---")
        print(f"{'Time':>8s}  {'Impr#':>6s}  {'BestCost':>12s}  {'k':>4s}")
        print("-" * 40)

    while not _time_is_up(start_time, time_limit):
        candidate_assignment = shake_insert(current_assignment, shake_strength)
        candidate_assignment = local_search(candidate_assignment, start_time, time_limit)
        candidate_cost, candidate_pm_slots = evaluate_schedule(decode_solution(candidate_assignment))

        if candidate_cost < best_cost:
            improvement_count += 1
            best_cost = candidate_cost
            best_assignment = candidate_assignment[:]
            best_pm_slots = candidate_pm_slots
            current_assignment = candidate_assignment[:]
            current_cost = candidate_cost
            stagnation = 0
            shake_strength = 1
            elapsed = time.time() - start_time
            history.append((elapsed, best_cost))

            if verbose:
                note = " [R]" if restart_triggered else ""
                print(f"{elapsed:8.2f}  {improvement_count:6d}  {best_cost:12.2f}  {shake_strength:4d}{note}")

            restart_triggered = False
        else:
            if candidate_cost < current_cost:
                current_assignment = candidate_assignment[:]
                current_cost = candidate_cost

            shake_strength = 1 if shake_strength >= SHAKE_MAX else shake_strength + 1
            stagnation += 1

        if stagnation >= MAX_STAGNATION:
            if verbose:
                elapsed = time.time() - start_time
                print(f"{elapsed:8.2f}  **RESTART** (stagnation={stagnation})")

            restart_assignment = generate_random_feasible_solution(start_time, time_limit)
            if restart_assignment is not None:
                restart_cost, restart_pm_slots = evaluate_schedule(decode_solution(restart_assignment))

                if restart_cost < best_cost:
                    best_cost = restart_cost
                    best_assignment = restart_assignment[:]
                    best_pm_slots = restart_pm_slots
                    improvement_count += 1
                    restart_triggered = True
                    elapsed = time.time() - start_time
                    history.append((elapsed, best_cost))

                    if verbose:
                        print(f"{elapsed:8.2f}  {improvement_count:6d}  {best_cost:12.2f}  {shake_strength:4d} [R]")

                current_assignment = restart_assignment[:]
                current_cost = restart_cost

            stagnation = 0
            shake_strength = 1

    total_elapsed = time.time() - start_time
    if history[-1][0] < total_elapsed:
        history.append((total_elapsed, best_cost))

    if verbose:
        print("=" * 40)

    return best_assignment, best_cost, best_pm_slots, history


def run_experiments(args: argparse.Namespace) -> None:
    """Run one or more independent LIM-VNS experiments."""
    config.validate_instance()
    time_limit = args.time_limit if args.time_limit is not None else default_time_limit()
    output_path = args.output or Path(__file__).with_name(f"lim_vns-{N}-{T}.txt")

    print_instance_summary()

    if args.runs == 1:
        random.seed(args.seed)
        print("Generating initial solution...")
        initial_schedule = random_initial_solution()
        initial_assignment = encode_solution(initial_schedule)
        initial_cost, _ = evaluate_schedule(initial_schedule)

        print("\n===== Initial Solution =====")
        print(f"Total cost: {initial_cost:.2f}")

        print("\n===== Running LIM-VNS =====")
        best_assignment, best_cost, _, history = simplified_vns(
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
        initial_assignment = encode_solution(random_initial_solution())
        best_assignment, best_cost, _, history = simplified_vns(
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


def _random_swap(assignment: Assignment) -> Assignment | None:
    job_1, job_2 = random.sample(range(N), 2)
    if assignment[job_1] == assignment[job_2]:
        return None

    new_assignment = assignment[:]
    new_assignment[job_1], new_assignment[job_2] = new_assignment[job_2], new_assignment[job_1]
    return new_assignment


def _random_insert(assignment: Assignment) -> Assignment | None:
    job = random.randint(0, N - 1)
    old_slot = assignment[job]
    candidate_slots = [slot for slot in range(1, T + 1) if slot != old_slot]

    if not candidate_slots:
        return None

    new_assignment = assignment[:]
    new_assignment[job] = random.choice(candidate_slots)
    return new_assignment


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
