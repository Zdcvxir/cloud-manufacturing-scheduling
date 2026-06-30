"""Exact mixed-integer programming model built with Gurobi.

The model follows the formula numbering used in the paper. It is intentionally
kept separate from the VNS heuristics because the default benchmark instance is
large and the exact model can be expensive to build and solve.
"""

from __future__ import annotations

import argparse
from typing import Any

import config

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - depends on a licensed external solver.
    gp = None
    GRB = None


def require_gurobi() -> None:
    """Raise a helpful error if gurobipy is unavailable."""
    if gp is None or GRB is None:
        raise RuntimeError(
            "gurobipy is required to build the exact model. Install Gurobi and "
            "activate a valid license before running gurobi_model.py."
        )


def build_model(time_limit: float = 3600.0) -> Any:
    """Build and return the Gurobi MIP model."""
    require_gurobi()
    config.validate_instance()

    n_jobs = config.N
    n_slots = config.T
    max_batches = config.B
    learning_exponent = config.a
    setup_time = config.seta
    failure_cost_weight = config.w
    pm_setup_cost = config.u
    failure_scale = config.k
    failure_rate = config.lam
    batch_capacity = config.c
    slot_processing_costs = config.tao_l
    processing_times = config.p_j
    slot_lengths = config.L_l
    slot_fixed_costs = config.CP_l

    model = gp.Model("batch_scheduling_mip")
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    # Binary variables
    X = model.addVars(max_batches, n_slots, n_jobs, vtype=GRB.BINARY, name="X")
    Y = model.addVars(max_batches, n_slots, vtype=GRB.BINARY, name="Y")
    Z = model.addVars(n_slots, vtype=GRB.BINARY, name="Z")
    x = model.addVars(n_jobs, n_jobs, vtype=GRB.BINARY, name="x")
    U = model.addVars(max_batches, n_slots, n_jobs, vtype=GRB.BINARY, name="U")
    Q = model.addVars(n_slots - 1, vtype=GRB.BINARY, name="Q")

    # Continuous variables
    P_bl = model.addVars(max_batches, n_slots, lb=0.0, vtype=GRB.CONTINUOUS, name="P_bl")
    P_l = model.addVars(n_slots, lb=0.0, vtype=GRB.CONTINUOUS, name="P_l")
    E_l = model.addVars(n_slots, lb=0.0, vtype=GRB.CONTINUOUS, name="E_l")
    G_l = model.addVars(n_slots, lb=0.0, vtype=GRB.CONTINUOUS, name="G_l")
    V_blr = model.addVars(max_batches, n_slots, n_jobs, lb=0.0, vtype=GRB.CONTINUOUS, name="V_blr")
    t_ls = model.addVars(n_slots, lb=0.0, vtype=GRB.CONTINUOUS, name="t_ls")
    t_le = model.addVars(n_slots, lb=0.0, vtype=GRB.CONTINUOUS, name="t_le")

    prefix_slot_lengths = [0.0]
    for slot_length in slot_lengths:
        prefix_slot_lengths.append(prefix_slot_lengths[-1] + float(slot_length))

    processing_coefficients = [
        [
            processing_times[job] * ((position + 1) ** learning_exponent)
            for position in range(n_jobs)
        ]
        for job in range(n_jobs)
    ]
    max_position_processing = [
        max(processing_coefficients[job][position] for job in range(n_jobs))
        for position in range(n_jobs)
    ]
    max_machine_age = [prefix_slot_lengths[slot] for slot in range(n_slots)]
    max_expected_cm_cost = [
        failure_scale * failure_rate * failure_cost_weight * prefix_slot_lengths[slot + 1]
        for slot in range(n_slots)
    ]
    big_m = max(
        max(max_position_processing),
        max(max_machine_age),
        max(max_expected_cm_cost),
    )

    for slot in range(n_slots):
        P_l[slot].UB = slot_lengths[slot]
        t_ls[slot].UB = prefix_slot_lengths[slot]
        t_le[slot].UB = prefix_slot_lengths[slot + 1]
        E_l[slot].UB = max_expected_cm_cost[slot]
        G_l[slot].UB = max_expected_cm_cost[slot]

    for batch in range(max_batches):
        for slot in range(n_slots):
            P_bl[batch, slot].UB = slot_lengths[slot]
            for position in range(n_jobs):
                V_blr[batch, slot, position].UB = max_position_processing[position]

    # Formula (3): objective function
    model.setObjective(
        gp.quicksum(
            slot_processing_costs[slot] * P_l[slot]
            + slot_fixed_costs[slot] * Z[slot]
            + G_l[slot]
            for slot in range(n_slots)
        )
        + gp.quicksum(pm_setup_cost * Q[slot] for slot in range(n_slots - 1)),
        GRB.MINIMIZE,
    )

    # Formula (4): each job is assigned to exactly one batch and one time slot.
    for job in range(n_jobs):
        model.addConstr(
            gp.quicksum(X[batch, slot, job] for batch in range(max_batches) for slot in range(n_slots)) == 1
        )

    # Formula (5): capacity limit of each batch in each time slot.
    for batch in range(max_batches):
        for slot in range(n_slots):
            model.addConstr(gp.quicksum(X[batch, slot, job] for job in range(n_jobs)) <= batch_capacity)

    # Formula (6): each position is occupied by exactly one job.
    for position in range(n_jobs):
        model.addConstr(gp.quicksum(x[job, position] for job in range(n_jobs)) == 1)

    # Formula (7): each job is assigned to exactly one position.
    for job in range(n_jobs):
        model.addConstr(gp.quicksum(x[job, position] for position in range(n_jobs)) == 1)

    # Formula (8): each position is assigned to exactly one batch-time-slot pair.
    for position in range(n_jobs):
        model.addConstr(
            gp.quicksum(
                U[batch, slot, position]
                for batch in range(max_batches)
                for slot in range(n_slots)
            )
            == 1
        )

    # Formula (9): link occupied positions with assigned jobs in each pair.
    for batch in range(max_batches):
        for slot in range(n_slots):
            model.addConstr(
                gp.quicksum(U[batch, slot, position] for position in range(n_jobs))
                == gp.quicksum(X[batch, slot, job] for job in range(n_jobs))
            )

    # Formula (10): continuity of positions within the same batch-time-slot pair.
    for batch in range(max_batches):
        for slot in range(n_slots):
            for position in range(1, n_jobs - 1):
                model.addConstr(
                    U[batch, slot, position]
                    >= U[batch, slot, position - 1] + U[batch, slot, position + 1] - 1
                )

    # Formula (11): link X, U, and x.
    for batch in range(max_batches):
        for slot in range(n_slots):
            for job in range(n_jobs):
                for position in range(n_jobs):
                    model.addConstr(
                        X[batch, slot, job] >= U[batch, slot, position] + x[job, position] - 1
                    )

    # Formula (12): link U, X, and x.
    for batch in range(max_batches):
        for slot in range(n_slots):
            for job in range(n_jobs):
                for position in range(n_jobs):
                    model.addConstr(
                        U[batch, slot, position] >= X[batch, slot, job] + x[job, position] - 1
                    )

    # Formula (13): keep the time-slot index nondecreasing with position.
    for position in range(n_jobs - 1):
        model.addConstr(
            gp.quicksum(
                (slot + 1) * gp.quicksum(U[batch, slot, position] for batch in range(max_batches))
                for slot in range(n_slots)
            )
            <= gp.quicksum(
                (slot + 1) * gp.quicksum(U[batch, slot, position + 1] for batch in range(max_batches))
                for slot in range(n_slots)
            )
        )

    # Formula (14): define whether batch b in time slot l is occupied.
    for batch in range(max_batches):
        for slot in range(n_slots):
            model.addConstr(
                Y[batch, slot]
                >= gp.quicksum(X[batch, slot, job] for job in range(n_jobs)) / batch_capacity
            )

    position_processing = [
        gp.quicksum(
            processing_coefficients[job][position] * x[job, position]
            for job in range(n_jobs)
        )
        for position in range(n_jobs)
    ]

    # Formula (15): preserve cumulative position order across time slots.
    for position in range(n_jobs - 1):
        for slot in range(n_slots - 1):
            model.addConstr(
                gp.quicksum(
                    U[batch, previous_slot, position]
                    for previous_slot in range(slot + 1)
                    for batch in range(max_batches)
                )
                >= gp.quicksum(
                    U[batch, previous_slot, position + 1]
                    for previous_slot in range(slot + 1)
                    for batch in range(max_batches)
                )
            )

    # Formulas (16)-(17): position-dependent processing time contribution.
    for batch in range(max_batches):
        for slot in range(n_slots):
            for position in range(n_jobs):
                model.addConstr(V_blr[batch, slot, position] <= position_processing[position])
                model.addConstr(
                    V_blr[batch, slot, position]
                    >= position_processing[position] - big_m * (1 - U[batch, slot, position])
                )

    # Formula (18): actual processing time of each batch.
    for batch in range(max_batches):
        for slot in range(n_slots):
            model.addConstr(
                P_bl[batch, slot]
                >= setup_time * Y[batch, slot]
                + gp.quicksum(V_blr[batch, slot, position] for position in range(n_jobs))
            )

    # Formula (19): actual processing time of each time slot.
    for slot in range(n_slots):
        model.addConstr(P_l[slot] >= gp.quicksum(P_bl[batch, slot] for batch in range(max_batches)))

    # Formula (20): length limit of each time slot.
    for slot in range(n_slots):
        model.addConstr(P_l[slot] <= slot_lengths[slot])

    # Formulas (21)-(24): machine age and preventive maintenance reset.
    model.addConstr(t_ls[0] == 0)
    for slot in range(n_slots):
        model.addConstr(t_le[slot] == t_ls[slot] + P_l[slot])

    for slot in range(1, n_slots):
        model.addConstr(t_ls[slot] <= t_le[slot - 1])
        model.addConstr(t_ls[slot] <= big_m * (1 - Q[slot - 1]))
        model.addConstr(t_ls[slot] >= t_le[slot - 1] - big_m * Q[slot - 1])

    # Formula (25): expected corrective-maintenance cost at ending machine age.
    for slot in range(n_slots):
        model.addConstr(E_l[slot] == failure_scale * failure_rate * failure_cost_weight * t_le[slot])

    # Formulas (26)-(28): linearization for G_l = E_l Z_l.
    for slot in range(n_slots):
        model.addConstr(G_l[slot] <= E_l[slot])
        model.addConstr(G_l[slot] <= big_m * Z[slot])
        model.addConstr(G_l[slot] >= E_l[slot] - big_m * (1 - Z[slot]))

    # Formula (29): define whether time slot l is selected.
    for slot in range(n_slots):
        model.addConstr(
            Z[slot] >= gp.quicksum(Y[batch, slot] for batch in range(max_batches)) / max_batches
        )

    # Formulas (30)-(31): binary and nonnegative domains are set by variable declarations.
    return model


def solve_model(time_limit: float, print_solution: bool = False) -> None:
    """Build, solve, and summarize the exact model."""
    model = build_model(time_limit=time_limit)
    print(f"Gurobi version: {gp.gurobi.version()}")
    print("Model built. Starting optimization...")
    model.optimize()

    if model.SolCount > 0:
        print(f"Best objective value: {model.ObjVal:.6f}")
        if print_solution:
            for variable in model.getVars():
                if abs(variable.X) > 1e-9:
                    print(f"{variable.VarName} = {variable.X}")
    else:
        print("No feasible solution was found within the configured time limit.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=3600.0, help="Gurobi time limit in seconds")
    parser.add_argument("--print-solution", action="store_true", help="print nonzero variable values")
    parser.add_argument("--build-only", action="store_true", help="build the model without optimizing")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.time_limit < 0:
        raise SystemExit("--time-limit must be non-negative.")

    if args.build_only:
        model = build_model(time_limit=args.time_limit)
        print(f"Model built with {model.NumVars} variables and {model.NumConstrs} constraints.")
        return

    solve_model(
        time_limit=args.time_limit,
        print_solution=args.print_solution,
    )


if __name__ == "__main__":
    main()
