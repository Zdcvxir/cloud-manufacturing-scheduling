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
    kr = model.addVars(n_jobs, lb=0.0, vtype=GRB.CONTINUOUS, name="kr")

    # Construct a unified but data-dependent Big-M
    prefix_slot_lengths = [0.0]
    for slot_length in slot_lengths:
        prefix_slot_lengths.append(prefix_slot_lengths[-1] + float(slot_length))

    # Upper bound of V_blr
    max_position_processing = [
        max(
            processing_times[job] * ((position + 1) ** learning_exponent)
            for job in range(n_jobs)
        )
        for position in range(n_jobs)
    ]

    # Upper bound of machine age
    max_machine_age = [prefix_slot_lengths[slot] for slot in range(n_slots)]

    # Upper bound of E_l
    # Current model: E_l[l] = k * lam * w * t_le[l]
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

    # 1. Formula (3): objective function
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

    # 2. Formula (4): each job is assigned to exactly one batch and one time slot
    for job in range(n_jobs):
        model.addConstr(
            gp.quicksum(X[batch, slot, job] for batch in range(max_batches) for slot in range(n_slots)) == 1
        )

    # 3. Formula (5): capacity limit of each batch in each time slot
    for batch in range(max_batches):
        for slot in range(n_slots):
            model.addConstr(gp.quicksum(X[batch, slot, job] for job in range(n_jobs)) <= batch_capacity)

    # 4. Formula (6): each position is occupied by exactly one job
    for position in range(n_jobs):
        model.addConstr(gp.quicksum(x[job, position] for job in range(n_jobs)) == 1)

    # 5. Formula (7): each job is assigned to exactly one position
    for job in range(n_jobs):
        model.addConstr(gp.quicksum(x[job, position] for position in range(n_jobs)) == 1)

    # 6. Formula (8): each position is assigned to exactly one batch and one time slot
    for position in range(n_jobs):
        model.addConstr(
            gp.quicksum(
                U[batch, slot, position]
                for batch in range(max_batches)
                for slot in range(n_slots)
            )
            == 1
        )

    # 7. Formula (9): link occupied positions with assigned jobs in each batch-time-slot pair
    for batch in range(max_batches):
        for slot in range(n_slots):
            model.addConstr(
                gp.quicksum(U[batch, slot, position] for position in range(n_jobs))
                == gp.quicksum(X[batch, slot, job] for job in range(n_jobs))
            )

    # 8. Formula (10): ordinal index of the batch-time-slot pair assigned to position r
    for position in range(n_jobs):
        model.addConstr(
            kr[position]
            == gp.quicksum(
                (slot * max_batches + batch + 1) * U[batch, slot, position]
                for slot in range(n_slots)
                for batch in range(max_batches)
            )
        )

    # 9. Formula (11): keep the batch-time-slot ordinal index nondecreasing with r
    for position in range(n_jobs - 1):
        model.addConstr(kr[position] <= kr[position + 1])

    # 10. Formula (12): link X, U, and x
    for batch in range(max_batches):
        for slot in range(n_slots):
            for job in range(n_jobs):
                for position in range(n_jobs):
                    model.addConstr(
                        X[batch, slot, job] >= U[batch, slot, position] + x[job, position] - 1
                    )

    # 11. Formula (13): link U, X, and x
    for batch in range(max_batches):
        for slot in range(n_slots):
            for job in range(n_jobs):
                for position in range(n_jobs):
                    model.addConstr(
                        U[batch, slot, position] >= X[batch, slot, job] + x[job, position] - 1
                    )

    # 12. Formula (14): define whether batch b in time slot l is occupied
    for batch in range(max_batches):
        for slot in range(n_slots):
            model.addConstr(
                Y[batch, slot]
                >= gp.quicksum(X[batch, slot, job] for job in range(n_jobs)) / batch_capacity
            )

    # Position-dependent processing coefficient p_j f(r), where f(r)=(r+1)^a in the code.
    processing_coefficients = [
        [
            processing_times[job] * ((position + 1) ** learning_exponent)
            for position in range(n_jobs)
        ]
        for job in range(n_jobs)
    ]
    position_processing = [
        gp.quicksum(
            processing_coefficients[job][position] * x[job, position]
            for job in range(n_jobs)
        )
        for position in range(n_jobs)
    ]

    # 13. Formula (15): first upper bound for V_blr
    for batch in range(max_batches):
        for slot in range(n_slots):
            for position in range(n_jobs):
                model.addConstr(
                    V_blr[batch, slot, position]
                    <= max_position_processing[position] * U[batch, slot, position]
                )
    # 14. Formula (16): second upper bound for V_blr
    for batch in range(max_batches):
        for slot in range(n_slots):
            for position in range(n_jobs):
                model.addConstr(V_blr[batch, slot, position] <= position_processing[position])

    # 15. Formula (17): lower bound for V_blr
    for batch in range(max_batches):
        for slot in range(n_slots):
            for position in range(n_jobs):
                model.addConstr(
                    V_blr[batch, slot, position]
                    >= position_processing[position] - big_m * (1 - U[batch, slot, position])
                )

    # 16. Formula (18): actual processing time of each batch
    for batch in range(max_batches):
        for slot in range(n_slots):
            model.addConstr(
                P_bl[batch, slot]
                >= setup_time * Y[batch, slot]
                + gp.quicksum(V_blr[batch, slot, position] for position in range(n_jobs))
            )

    # 17. Formula (19): actual processing time of each time slot
    for slot in range(n_slots):
        model.addConstr(P_l[slot] >= gp.quicksum(P_bl[batch, slot] for batch in range(max_batches)))

    # 18. Formula (20): length limit of each time slot
    for slot in range(n_slots):
        model.addConstr(P_l[slot] <= slot_lengths[slot])

    # 19. Formula (21): ending machine age and initial starting age
    model.addConstr(t_ls[0] == 0)
    for slot in range(n_slots):
        model.addConstr(t_le[slot] == t_ls[slot] + P_l[slot])

    # 20. Formula (22): machine age cannot exceed the previous ending age before reset
    for slot in range(1, n_slots):
        model.addConstr(t_ls[slot] <= t_le[slot - 1])

    # 21. Formula (23): reset starting age to zero if PM is performed
    for slot in range(1, n_slots):
        model.addConstr(t_ls[slot] <= big_m * (1 - Q[slot - 1]))

    # 22. Formula (24): inherit previous ending age if PM is not performed
    for slot in range(1, n_slots):
        model.addConstr(t_ls[slot] >= t_le[slot - 1] - big_m * Q[slot - 1])

    # 23. Formula (25): expected CM cost evaluated at the ending machine age
    for slot in range(n_slots):
        model.addConstr(E_l[slot] == failure_scale * failure_rate * failure_cost_weight * t_le[slot])

    # 24. Formula (26): first upper bound for G_l = E_l Z_l
    for slot in range(n_slots):
        model.addConstr(G_l[slot] <= E_l[slot])

    # 25. Formula (27): second upper bound for G_l = E_l Z_l
    for slot in range(n_slots):
        model.addConstr(G_l[slot] <= big_m * Z[slot])

    # 26. Formula (28): lower bound for G_l = E_l Z_l
    for slot in range(n_slots):
        model.addConstr(G_l[slot] >= E_l[slot] - big_m * (1 - Z[slot]))

    # 27. Formula (29): define whether time slot l is selected
    for slot in range(n_slots):
        model.addConstr(
            Z[slot] >= gp.quicksum(Y[batch, slot] for batch in range(max_batches)) / max_batches
        )

    # 28. Formula (30): binary domains
    # X, Y, Z, x, U, and Q are declared as GRB.BINARY above.

    # 29. Formula (31): nonnegative continuous domains
    # P_bl, P_l, E_l, t_ls, t_le, V_blr, G_l, and kr are declared with lb=0.0 above.
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
