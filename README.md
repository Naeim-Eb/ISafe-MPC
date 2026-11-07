# Simulation scripts (revised version)

Paper: "ISafe-MPC: Recursively Feasible Model Predictive Control with Intersample Safety
under Zero-Order Hold".

Requirements: Python 3.9+, `numpy`, `scipy`, `casadi` (ships with IPOPT), `sympy` (ex0 only).

    pip install numpy scipy casadi sympy

| Script | Paper section | Runtime (approx.) |
|---|---|---|
| `ex1_double_integrator.py` | IV-A (Example 1 and 1B), Table I, Fig. 3 | ~45 min (`A`: ~5 min, `B`: ~40 min; run `python3 ex1_double_integrator.py A` and `... B` in parallel) |
| `timing.py` | Table I, column t_solve of all examples (sequential timing, run last) | ~5 min |
| `ex2_pvtol.py` | IV-B, Table I, Fig. 4 | ~40 min |
| `ex0_unstable_di_sampling_time.py` | II-B Example 1 (unstable double integrator: CT-CI but not DT-CI) and III-B Example 1 (continued): T*, validation of Theorem 1 | ~3 min |
| `make_figure_data.py` | converts results/*.csv into the pgfplots coordinates of Figs. 3-4 | seconds |

Run the three example scripts, then `timing.py` and `make_figure_data.py`. All reported numbers are
written to `results/ex*_summary.txt`.

Method labels:
- F1   stage-wise discrete-time CBF on c (Zeng et al. 2021), gamma swept in Example 1
- F1h  stage-wise DCBF on the velocity-aware braking function h_b (Example 1)
- F2   terminal-set MPC (eq. (5) of rev10)
- F3   proposed MPC (eq. (12) of rev10); F3C = terminal set certified by Theorem 1 +
       Proposition 1 (Case 1B); F3SD = node-dependent remainder bound (Case 2; in rev10 it
       appears only in the "Sampling time" paragraph, "code online")
- F4   linear-system baseline (Example 1 only), in the spirit of Sopasakis et al. (TAC 2014)
       and Kim & Park (ACC 2024): the intersample constraint is imposed on 21 grid points
       per sampling interval with the certified tightening U_MAX (Ts/20)^2/8, which is
       exact for the double integrator up to O(1/20^2)

Solver notes: IPOPT runs with constraint tolerance 1e-8 in all examples and the same
"acceptable" tolerance; the IPOPT default acceptable tolerance (1e-2) lets slightly
infeasible solutions through. Initial guesses are always set explicitly (cold start
X = x0, U = 0 in Ex. 1 / hover in Ex. 2; afterwards the shifted solution), so results do
not depend on the order of the runs.

Solver failures: if IPOPT fails, the shifted candidate of the proof of Theorem 2 (inputs
u_1..u_{N-1} of the last solution followed by the compatible input of Definition 2 at the terminal
state: u = -1 if x2 > 0, else u = 0, for braking sets; u = -K(x - xe) for the certified LQR
ellipsoid of Example 1B; hover thrust in Example 2) is applied only after it has been
verified to satisfy all MPC constraints from the current state (tolerance CAND_TOL = 1e-7 in
Example 1 and 1e-6 in Example 2, where it covers the RK4 prediction error); this is a "fallback"
step, and the candidate warm-starts the next solve. Only formulations with a terminal
set (F2, F3, F3C, F4 / F2, F3, F3SD) have a candidate. A solver failure without a
verified candidate counts as a loss of feasibility ("lost feas." in Table I).

Data resolution: Example 2 trajectories are stored at full resolution (25 points per
sampling interval, 1e-8 m), and the interval minima of Fig. 4(b) are computed from them
in ex2_pvtol.py (results/ex2_interval_min_<F>.csv).

Closed-loop cost (Table I): sum_k Ts * l(x(t_k), u_k) in Example 1/1B (ex1_double_integrator.py)
and sum_k l(x(t_k), u_k) in Example 2 (ex2_pvtol.py). "Example 1/1B/2" in the scripts are
"Case 1/1B/2" in the paper.
