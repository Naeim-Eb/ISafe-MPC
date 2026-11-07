#!/usr/bin/env python3
"""
Example 2 -- Planar VTOL (planar quadrotor) between a wall and the floor.

Reproduces Section IV-B of the paper "Intersample-Safe Model Predictive
Control with Terminal Invariant Sets and High-Order Sampled-Data Barrier
Constraints".

State  x = (y, z, theta, vy, vz, w)  (horizontal/vertical position, pitch,
velocities, pitch rate); input u = (F, tau) (total thrust, torque):
    y' = vy,  z' = vz,  theta' = w,
    vy' = -(F/m) sin(theta),  vz' = (F/m) cos(theta) - g,  w' = tau/J,
    0 <= F <= F_MAX,  |tau| <= TAU_MAX.
Admissible set (two constraints, both of relative degree r = 2 w.r.t. F):
    c1(x) = Y_WALL - y >= 0     (vertical wall)
    c2(x) = z - Z_FLOOR >= 0    (floor)
Unlike Example 1, the third Lie derivative is nonzero, so the Taylor
remainder bound M_{c,3} of the paper is active:
    L^3 c1 =  (F/m) cos(theta) w,   L^3 c2 = -(F/m) sin(theta) w,
    |L^3 c_j| <= F_MAX * W_K / m =: M3,  W_K = W_MAX + TAU_MAX * Ts / J,
where W_K bounds the pitch rate on the set K reachable within one sampling
interval from the node set X = {|w| <= W_MAX, ...} (Assumption 1).

Methods (same model, cost, horizon, solver; see ex1 for the definitions):
  F1    stage-wise DCBF on c1, c2 (no terminal set)
  F2    node constraints c_j(x_i) >= 0 + terminal hover set with c_j >= 0
  F3    Bernstein constraints beta_{c_j,nu}(x_i,u_i) >= 0, nu = 0..3, with
        the global remainder bound M3, + terminal hover set with
        c_j >= eta = M3 Ts^3 / 6
  F3SD  Bernstein constraints with the *node-dependent* remainder bound of
        Lemma 1(ii) of the paper: along a ZOH interval omega(s) = w_i +
        tau_i s / J, hence |L^3 c_j(xi(s))| <= (F_i/m)(|w_i| + TAU_MAX s/J)
        =: mu0 + mu1 s, giving the extra power coefficients
            a3 = -Ts^3/6 * F_i |w_i| / m,   a4 = -Ts^4/24 * F_i TAU_MAX/(m J)
        (degree-4 Bernstein form, still affine in F_i).  |w_i| is replaced by
        the smooth upper bound sqrt(w_i^2 + 1e-4) (|w| is not differentiable
        at hover, w = 0, which IPOPT needs).  At hover (w = 0, F = m g):
        a1 = a2 = 0, a3 = -g Ts^3 0.01/6 (0.016 mm), a4 = -g TAU_MAX Ts^4/(24 J)
        (1.02 mm), so the tightest coefficient is beta_4 = c - 1.04 mm; the
        terminal tightening is eta_SD = 2 * 1.04 mm = 2.08 mm (see eta_for).

Costs (identical for all methods):
  stage    l(x,u) = 10||(y,z)-p_ref||^2 + 0.1||(vy,vz,w)||^2 + 0.5 theta^2
                    + 0.01 (F - m g)^2 + tau^2
  terminal Vf(x)  = 10||(y,z)-p_ref||^2 + ||(vy,vz,w)||^2
The hover set {vy = vz = w = theta = 0} is sampled-data invariant under the
hover input (F, tau) = (m g, 0); at hover, a1 = a2 = 0 and
beta_{c,3} = c - M3 Ts^3/6, so c >= eta gives terminal compatibility
(Assumption 3) exactly.

The prediction model is the ZOH map computed with RK4 (10 substeps); the
plant is integrated with scipy's adaptive RK45 at tight tolerances and the
constraints are evaluated densely between samples.

Solver failures: as in ex1, after an IPOPT failure the shifted candidate of
the proof of Theorem 2 (completed with the hover input) is applied only if it
is verified to satisfy all MPC constraints from the current plant state
(tolerance CAND_TOL, which covers the RK4 prediction error); this is a
"fallback" step, and the candidate warm-starts the next solve.  F1 has no
terminal set and hence no candidate.  A step without a verified candidate is
a loss of feasibility.  The RK4 prediction is compared with the plant at every
step ("mismatch").

Outputs (./results/): ex2_closed_loop_<F>.csv (full-resolution nominal
trajectories), ex2_interval_min_<F>.csv (minimum of min(c1, c2) over each
sampling interval, used for Fig. 4(b)), ex2_summary.txt
Requirements: numpy, scipy, casadi.        Run: python3 ex2_pvtol.py
"""
import os
import time
from math import comb

import numpy as np
import casadi as ca
from scipy.integrate import solve_ivp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
M, G, J = 1.0, 9.81, 0.02
F_MAX, TAU_MAX = 2.0 * M * G, 0.5
Y_WALL, Z_FLOOR = 2.0, 0.0
TH_MAX, W_MAX, V_MAX = 0.6, 3.0, 4.0     # node set X
GAMMA = 0.3                              # DCBF rate for F1
METHODS = ["F1", "F2", "F3", "F3SD"]
NX, NU = 6, 2


def waypoint(t):
    """Waypoints placed *beyond* the constraints: first behind the wall,
    then below the floor, then behind the wall again."""
    if t < 4.0:
        return np.array([2.6, 1.5])
    if t < 8.0:
        return np.array([0.8, -0.5])
    return np.array([2.6, 0.6])


def f_cont(x, u, lib=np):
    """Continuous-time vector field (numpy or casadi)."""
    y, z, th, vy, vz, w = [x[i] for i in range(6)]
    F, tau = u[0], u[1]
    return [vy, vz, w, -(F / M) * lib.sin(th), (F / M) * lib.cos(th) - G, tau / J]


def rk4_zoh(Ts, n_sub=10):
    """CasADi function x_{k+1} = F(x_k, u_k): ZOH map via RK4."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    h = Ts / n_sub
    xk = x
    for _ in range(n_sub):
        k1 = ca.vertcat(*f_cont(xk, u, ca))
        k2 = ca.vertcat(*f_cont(xk + h / 2 * k1, u, ca))
        k3 = ca.vertcat(*f_cont(xk + h / 2 * k2, u, ca))
        k4 = ca.vertcat(*f_cont(xk + h * k3, u, ca))
        xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("F", [x, u], [xk])


def remainder_bound(Ts):
    """M_{c,3} for both constraints (see module docstring)."""
    w_K = W_MAX + TAU_MAX * Ts / J
    return F_MAX * w_K / M


def bernstein(a):
    """Bernstein coefficients (eq. (8)) from power coefficients a_0..a_d."""
    d = len(a) - 1
    return [sum(comb(k, j) / comb(d, j) * a[j] for j in range(k + 1)) for k in range(d + 1)]


ABS_EPS = 1e-4


def abs_smooth(v):
    """Smooth upper bound of |v|: sqrt(v^2 + eps) >= |v| (over-estimate
    <= sqrt(eps) = 0.01 rad/s).  eps = 1e-8 makes the NLP ill-conditioned."""
    return ca.sqrt(v**2 + ABS_EPS)


def bernstein_wall_sd(x, u, Ts):
    """Node-dependent remainder (F3SD) for c1 = Y_WALL - y."""
    a = [Y_WALL - x[0], -x[3] * Ts, 0.5 * Ts**2 * u[0] * ca.sin(x[2]) / M,
         -Ts**3 / 6 * u[0] * abs_smooth(x[5]) / M, -Ts**4 / 24 * u[0] * TAU_MAX / (M * J)]
    return bernstein(a)


def bernstein_floor_sd(x, u, Ts):
    """Node-dependent remainder (F3SD) for c2 = z - Z_FLOOR."""
    a = [x[1] - Z_FLOOR, x[4] * Ts, 0.5 * Ts**2 * (u[0] * ca.cos(x[2]) / M - G),
         -Ts**3 / 6 * u[0] * abs_smooth(x[5]) / M, -Ts**4 / 24 * u[0] * TAU_MAX / (M * J)]
    return bernstein(a)


def eta_for(method, Ts):
    """Terminal tightening that makes the hover set compatible (Assumption 3)."""
    if method == "F3":
        return remainder_bound(Ts) * Ts**3 / 6
    if method == "F3SD":
        # minimal value at hover (w = 0, F = m g): -(a3 + a4) with the smoothed
        # |w| = sqrt(ABS_EPS); any larger eta keeps Assumption 3.  We use twice
        # the minimum so that, when the drone parks at c = eta, the Bernstein
        # constraints (which all equal c at hover) are not simultaneously active.
        eta_min = G * TAU_MAX * Ts**4 / (24 * J) + G * np.sqrt(ABS_EPS) * Ts**3 / 6
        return 2.0 * eta_min
    return 0.0


def bernstein_wall(x, u, Ts, M3, lib=ca):
    """beta_{c1,k}: c1 = Y_WALL - y; a1 = -vy Ts, a2 = Ts^2/2 * F sin(th)/m."""
    a = [Y_WALL - x[0], -x[3] * Ts, 0.5 * Ts**2 * u[0] * lib.sin(x[2]) / M,
         -Ts**3 / 6 * M3]
    return bernstein(a)


def bernstein_floor(x, u, Ts, M3, lib=ca):
    """beta_{c2,k}: c2 = z - Z_FLOOR; a1 = vz Ts, a2 = Ts^2/2 (F cos(th)/m - g)."""
    a = [x[1] - Z_FLOOR, x[4] * Ts, 0.5 * Ts**2 * (u[0] * lib.cos(x[2]) / M - G),
         -Ts**3 / 6 * M3]
    return bernstein(a)


# --------------------------------------------------------------------------
# MPC
# --------------------------------------------------------------------------
def build_mpc(method, N, Ts):
    Fd = rk4_zoh(Ts)
    M3 = remainder_bound(Ts)
    eta = eta_for(method, Ts)
    opti = ca.Opti()
    X = opti.variable(NX, N + 1)
    U = opti.variable(NU, N)
    x0 = opti.parameter(NX)
    ref = opti.parameter(2, N + 1)
    opti.subject_to(X[:, 0] == x0)
    cost = 0
    for i in range(N):
        xi, ui = X[:, i], U[:, i]
        opti.subject_to(X[:, i + 1] == Fd(xi, ui))
        opti.subject_to(opti.bounded(0, ui[0], F_MAX))
        opti.subject_to(opti.bounded(-TAU_MAX, ui[1], TAU_MAX))
        # node set X (Assumption 1)
        opti.subject_to(opti.bounded(-TH_MAX, X[2, i + 1], TH_MAX))
        opti.subject_to(opti.bounded(-W_MAX, X[5, i + 1], W_MAX))
        opti.subject_to(opti.bounded(-V_MAX, X[3:5, i + 1], V_MAX))
        cost += 10 * ca.sumsqr(xi[0:2] - ref[:, i]) + 0.1 * ca.sumsqr(xi[3:6]) \
            + 0.5 * xi[2]**2 + 0.01 * (ui[0] - M * G)**2 + 1.0 * ui[1]**2
        if method == "F1":
            for c_i, c_n in ((Y_WALL - xi[0], Y_WALL - X[0, i + 1]),
                             (xi[1] - Z_FLOOR, X[1, i + 1] - Z_FLOOR)):
                opti.subject_to(c_n >= (1 - GAMMA) * c_i)
        elif method == "F2":
            if i >= 1:
                opti.subject_to(Y_WALL - xi[0] >= 0)
                opti.subject_to(xi[1] - Z_FLOOR >= 0)
        elif method == "F3":
            for b in bernstein_wall(xi, ui, Ts, M3) + bernstein_floor(xi, ui, Ts, M3):
                opti.subject_to(b >= 0)
        elif method == "F3SD":
            for b in bernstein_wall_sd(xi, ui, Ts) + bernstein_floor_sd(xi, ui, Ts):
                opti.subject_to(b >= 0)
    xN = X[:, N]
    cost += 10 * ca.sumsqr(xN[0:2] - ref[:, N]) + 1.0 * ca.sumsqr(xN[3:6])
    if method in ("F2", "F3", "F3SD"):
        # terminal hover set: sampled-data invariant under (m g, 0)
        opti.subject_to(xN[2:6] == 0)
        opti.subject_to(Y_WALL - xN[0] >= eta)
        opti.subject_to(xN[1] - Z_FLOOR >= eta)
    opti.minimize(cost)
    # strict tolerances: IPOPT's default "acceptable" constraint violation
    # (1e-2) would otherwise let slightly infeasible solutions through
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
                          "ipopt.max_iter": 1000, "ipopt.tol": 1e-7,
                          "ipopt.mu_strategy": "adaptive",
                          "ipopt.constr_viol_tol": 1e-8,
                          "ipopt.acceptable_constr_viol_tol": 1e-8})
    # numeric stage constraints (all >= 0), used to verify shifted candidates
    xs_, us_ = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
    if method == "F3":
        g = bernstein_wall(xs_, us_, Ts, M3) + bernstein_floor(xs_, us_, Ts, M3)
    elif method == "F3SD":
        g = bernstein_wall_sd(xs_, us_, Ts) + bernstein_floor_sd(xs_, us_, Ts)
    else:
        g = [Y_WALL - xs_[0], xs_[1] - Z_FLOOR]
    stage_g = ca.Function("g", [xs_, us_], [ca.vertcat(*g)])
    return dict(opti=opti, X=X, U=U, x0=x0, ref=ref, N=N, eta=eta, M3=M3,
                method=method, Fd=Fd, stage_g=stage_g, Ts=Ts)


HOVER = np.array([M * G, 0.0])
CAND_TOL = 1e-6      # covers the RK4 prediction error (<= 1e-7 in all runs)


def candidate_states(mpc, x, useq):
    xs = [np.asarray(x, float)]
    for u in useq:
        xs.append(np.array(mpc["Fd"](xs[-1], u)).ravel())
    return np.array(xs).T


def candidate_ok(mpc, x, useq):
    """Verify that the input sequence useq satisfies all MPC constraints of
    build_mpc from the state x (tolerance CAND_TOL)."""
    method, tol, eta = mpc["method"], CAND_TOL, mpc["eta"]
    if method not in ("F2", "F3", "F3SD"):
        return False
    xs = candidate_states(mpc, x, useq)
    for i, u in enumerate(useq):
        if not (-tol <= u[0] <= F_MAX + tol and abs(u[1]) <= TAU_MAX + tol):
            return False
        xn = xs[:, i + 1]
        if abs(xn[2]) > TH_MAX + tol or abs(xn[5]) > W_MAX + tol or np.any(np.abs(xn[3:5]) > V_MAX + tol):
            return False
        if method == "F2" and i == 0:
            continue                             # node constraint only for i >= 1
        if np.array(mpc["stage_g"](xs[:, i], u)).min() < -tol:
            return False
    xN = xs[:, -1]
    return (np.all(np.abs(xN[2:6]) <= tol) and Y_WALL - xN[0] >= eta - tol
            and xN[1] - Z_FLOOR >= eta - tol)


def plant_interval(x, u, Ts, n_dense):
    """High-accuracy integration of the plant over one ZOH interval."""
    sol = solve_ivp(lambda t, s: f_cont(s, u), (0.0, Ts), x, method="RK45",
                    rtol=1e-10, atol=1e-12, dense_output=True)
    ts = np.linspace(0.0, Ts, n_dense, endpoint=False)
    return sol.y[:, -1], ts, sol.sol(ts)


def closed_loop(method, Ts, N, t_final=12.0, x_init=None, n_dense=25, mpc=None):
    mpc = build_mpc(method, N, Ts) if mpc is None else mpc
    opti, X, U = mpc["opti"], mpc["X"], mpc["U"]
    method = mpc["method"]
    x = np.array([0.0, 1.0, 0, 0, 0, 0]) if x_init is None else np.array(x_init, float)
    steps = int(round(t_final / Ts))
    rows, n_infeas, tsolve, first_infeas = [], 0, [], None
    n_fallback = 0      # solver failures handled with a verified shifted candidate
    cand = None         # shifted candidate (proof of Theorem 2): remaining inputs
    mismatch = 0.0      # max ||plant x(t_{k+1}) - predicted x_1^*|| (RK4 vs. plant)
    x1_pred = None
    max_viol, cost = 0.0, 0.0
    Xw, Uw = None, None
    for k in range(steps):
        t = k * Ts
        opti.set_value(mpc["x0"], x)
        opti.set_value(mpc["ref"], np.column_stack([waypoint(t + i * Ts) for i in range(N + 1)]))
        if Xw is not None:
            opti.set_initial(X, Xw)
            opti.set_initial(U, Uw)
        else:
            opti.set_initial(X, np.tile(x[:, None], (1, N + 1)))
            opti.set_initial(U, np.tile([[M * G], [0.0]], (1, N)))
        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            Uopt, Xopt = sol.value(U), sol.value(X)
            u = Uopt[:, 0]
            x1_pred = Xopt[:, 1]
            # warm start = shifted solution completed with the hover input; for
            # F2/F3/F3SD this is the candidate of the proof of Theorem 2
            Xw = np.column_stack([Xopt[:, 1:], Xopt[:, -1]])
            Uw = np.column_stack([Uopt[:, 1:], HOVER])
            cand = [Uopt[:, j] for j in range(1, N)] + [HOVER] if method != "F1" else None
        except RuntimeError:
            x1_pred = None
            if cand is not None and candidate_ok(mpc, x, cand):
                # Solver failure while a verified feasible point is known: apply
                # the shifted candidate (suboptimal MPC) and warm-start with it.
                n_fallback += 1
                xs_c = candidate_states(mpc, x, cand)
                u = cand[0]
                cand = cand[1:] + [HOVER]
                Xw = np.column_stack([xs_c[:, 1:], xs_c[:, -1]])
                Uw = np.column_stack(cand)
            else:
                n_infeas += 1
                first_infeas = k if first_infeas is None else first_infeas
                u = HOVER.copy()             # no verified candidate: hover thrust
                cand, Xw, Uw = None, None, None
        tsolve.append(time.perf_counter() - t0)
        x_next, ts, xs = plant_interval(x, u, Ts, n_dense)
        if x1_pred is not None:
            mismatch = max(mismatch, float(np.linalg.norm(x_next - x1_pred)))
        c_wall, c_floor = Y_WALL - xs[0], xs[1] - Z_FLOOR
        max_viol = max(max_viol, -c_wall.min(), -c_floor.min())
        for j in range(len(ts)):
            rows.append((t + ts[j], xs[0, j], xs[1, j], xs[2, j], u[0], u[1]))
        wp = waypoint(t)
        cost += 10 * np.sum((x[:2] - wp)**2) + 0.1 * np.sum(x[3:]**2) + 0.5 * x[2]**2 \
            + 0.01 * (u[0] - M * G)**2 + u[1]**2
        x = x_next
    return dict(rows=np.array(rows), n_infeas=n_infeas, first_infeas=first_infeas,
                mismatch=mismatch, n_fallback=n_fallback,
                max_viol=max(max_viol, 0.0),
                cost=cost * Ts, t_solve_ms=1e3 * np.median(tsolve))


def initially_feasible(mpc, x0, Ts):
    """Is the MPC feasible at t = 0 from x0?  (Theorem 2 assumes this.)"""
    opti, N = mpc["opti"], mpc["N"]
    opti.set_value(mpc["x0"], x0)
    opti.set_value(mpc["ref"], np.column_stack([waypoint(i * Ts) for i in range(N + 1)]))
    opti.set_initial(mpc["X"], np.tile(x0[:, None], (1, N + 1)))
    opti.set_initial(mpc["U"], np.tile([[M * G], [0.0]], (1, N)))
    try:
        opti.solve()
        return True
    except RuntimeError:
        return False


def monte_carlo(method, Ts, N, n_runs=60, t_final=12.0, seed=3):
    """Random initial positions/velocities near the corner (hover attitude).
    The same sample is used for every method; a run is counted only if the
    method is feasible at t = 0.  Returns (#feasible starts, #runs with an
    intersample violation, #runs that later lost feasibility, worst viol)."""
    rng = np.random.default_rng(seed)
    X0 = [np.array([rng.uniform(0.0, 1.8), rng.uniform(0.2, 2.0), 0.0,
                    rng.uniform(-1.0, 2.0), rng.uniform(-2.0, 1.0), 0.0])
          for _ in range(n_runs)]
    mpc = build_mpc(method, N, Ts)
    n_ok = n_viol = n_lost = n_fb = 0
    worst = 0.0
    for x0 in X0:
        if not initially_feasible(mpc, x0, Ts):
            continue
        r = closed_loop(method, Ts, N, t_final=t_final, x_init=x0, n_dense=15, mpc=mpc)
        n_ok += 1
        n_viol += r["max_viol"] > 1e-6
        n_lost += r["n_infeas"] > 0
        n_fb += r["n_fallback"]
        worst = max(worst, r["max_viol"])
    return n_ok, n_viol, n_lost, worst, n_fb


def steps_of(t_final, Ts):
    return int(round(t_final / Ts))


# --------------------------------------------------------------------------
def main():
    Ts, N = 0.1, 20
    M3 = remainder_bound(Ts)
    lines = [f"Example 2: PVTOL, Ts = {Ts}, N = {N}, M3 = {M3:.2f}, "
             f"eta(F3) = {eta_for('F3', Ts) * 1e3:.2f} mm, eta(F3SD) = {eta_for('F3SD', Ts) * 1e3:.3f} mm"]
    lines.append("\nNominal closed loop (12 s, waypoints beyond wall/floor):")
    lines.append(f"{'method':>6} {'viol[mm]':>9} {'#infeas':>8} {'#fallback':>9} {'cost':>8} {'t_solve[ms]':>11} "
                 f"{'min margin[mm]':>14} {'RK4 mismatch':>13}")
    for m in METHODS:
        r = closed_loop(m, Ts, N)
        rows = r["rows"]
        # full resolution (25 points per interval), 1e-8 m precision
        np.savetxt(os.path.join(OUT, f"ex2_closed_loop_{m}.csv"), rows, delimiter=",",
                   header="t,y,z,theta,F,tau", comments="", fmt="%.8f")
        # Fig. 4(b): minimum of min(c1, c2) over each sampling interval
        mg = np.minimum(Y_WALL - rows[:, 1], rows[:, 2] - Z_FLOOR)
        kk = np.floor(rows[:, 0] / Ts + 1e-9).astype(int)
        mins = np.array([mg[kk == j].min() for j in range(steps_of(12.0, Ts))])
        np.savetxt(os.path.join(OUT, f"ex2_interval_min_{m}.csv"),
                   np.column_stack([np.arange(len(mins)) * Ts, 1e3 * mins]), delimiter=",",
                   header="t_k,min_margin_mm", comments="", fmt="%.4f")
        margin = np.minimum(Y_WALL - rows[:, 1], rows[:, 2] - Z_FLOOR).min()
        lines.append(f"{m:>6} {1e3 * r['max_viol']:9.3f} {r['n_infeas']:8d} {r['n_fallback']:9d} {r['cost']:8.2f} "
                     f"{r['t_solve_ms']:11.1f} {1e3 * margin:14.2f} {r['mismatch']:13.2e}")

    lines.append("\nMonte Carlo (60 random x0, 12 s; only starts feasible at t=0 are run):")
    lines.append(f"{'method':>6} {'#feas x0':>9} {'#viol runs':>11} {'#lost feas':>11} {'worst[mm]':>10} {'#fallback steps':>16}")
    for m in METHODS:
        n, nv, nl, w, nfb = monte_carlo(m, Ts, N)
        lines.append(f"{m:>6} {n:9d} {nv:11d} {nl:11d} {1e3 * w:10.3f} {nfb:16d}")

    lines.append("\nSampling-time sweep (horizon 2 s): viol[mm] / #infeasible / cost / #fallback steps")
    for Tsw in (0.05, 0.1, 0.2):
        Nw = int(round(2.0 / Tsw))
        row = [f"Ts={Tsw:4.2f}"]
        for m in METHODS:
            r = closed_loop(m, Tsw, Nw)
            row.append(f"{m}: {1e3 * r['max_viol']:.3f}/{r['n_infeas']}/{r['cost']:.2f}/fb{r['n_fallback']}")
        row.append(f"eta(F3)={1e3 * eta_for('F3', Tsw):.1f}mm eta(F3SD)={1e3 * eta_for('F3SD', Tsw):.2f}mm")
        lines.append("  ".join(row))

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "ex2_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
