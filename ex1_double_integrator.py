#!/usr/bin/env python3
"""
Example 1 -- Double integrator approaching a wall (relative degree r = 2).

Reproduces every number and data file of Section IV-A of the paper
"Intersample-Safe Model Predictive Control with Terminal Invariant Sets and
High-Order Sampled-Data Barrier Constraints".

System (continuous time, zero-order hold with period Ts):
    x1' = x2,   x2' = u,   |u| <= 1,   |x2| <= V_MAX at the nodes
Admissible set:  A = { x : c(x) = 1 - x1 >= 0 }   (a wall at x1 = 1).
c has input-free derivatives up to order r - 1 = 1 and L^3_{F_u} c = 0, so the
Taylor polynomial P_c of the paper is exact (M_{c,3} = 0) and the ZOH model
below is exact.

Formulations (identical model, cost, horizon, solver):
  F1   stage-wise DCBF on c (Zeng et al., ACC 2021): c(x_{i+1}) >= (1-g) c(x_i)
  F1h  stage-wise DCBF on the velocity-aware braking function
       h_b(x) = 1 - x1 - max(x2,0)^2/2 (zero superlevel set = continuous-time
       viability kernel of A), as suggested for a "valid" DCBF
  F2   terminal-set MPC: node constraints c(x_i) >= 0, terminal braking set
       S_1 = {x1 + max(x2,0)^2/2 <= 1}
  F3   proposed: Bernstein constraints beta_{c,nu}(x_i,u_i) >= 0 (nu = 0..3)
       + tightened braking set S_eta, xbar = 1 - Ts^2/18 (direct check of
       Assumption 3, see check_terminal_compatibility)
  F3C  proposed with the *constructive* route of the paper: terminal set is
       an LQR ellipsoid S_L = {(x-xe)' P (x-xe) <= rho}, certified by
       Theorem 1 (explicit sampling time) and Corollary 1 (tightening eta_c);
       see design_certified_terminal_set().
  F4   linear-system baseline in the spirit of Sopasakis et al. (TAC 2014) and
       Kim & Park (ACC 2024): the intersample constraint is imposed on a grid of
       M_GRID + 1 points per sampling interval, c(xi(j Ts/M_GRID; x_i, u_i)) >=
       eps_m, j = 0..M_GRID, which is linear in (x_i, u_i).  For the double
       integrator, x1(t) is a quadratic in t with |x1''| <= U_MAX, so between two
       grid points it exceeds the chord by at most U_MAX (Ts/M_GRID)^2 / 8 =:
       eps_m; the gridded constraint is therefore *certified* and becomes exact
       as M_GRID grows.  Terminal set: braking set with xbar = 1 - eps_m.  This
       construction exploits linearity and is not available for Example 2.

Experiments:
  (A) main setting Ts = 1, N = 8: nominal run, Monte Carlo (300 x0), feasible
      sets, Ts sweep, and a gamma sweep for F1 (fairness of the baseline)
  (B) constructive pipeline Ts = 0.1, N = 80: F2 vs F3 vs F3C vs F4

Solver failures: if IPOPT fails, the shifted candidate of the proof of
Theorem 2 is applied (suboptimal MPC, Scokaert et al. 1999) *after* it has been
verified to satisfy all MPC constraints from the current state, and the step
is counted as a "fallback"; the rest of the candidate warm-starts the next
solve.  Only formulations with a terminal set (F2, F3, F3C, F4) have such a
candidate.  A step without a verified candidate counts as a loss of
feasibility.  All initial guesses are set explicitly at every solve (cold
start: X = x0, U = 0), so results do not depend on the order of the runs.

Outputs (./results/): ex1_closed_loop_<F>.csv, ex1_feasible_boundary.csv,
ex1_summary.txt.        Requirements: numpy, scipy, casadi.
Run: python3 ex1_double_integrator.py          (all, ~60 min)
     python3 ex1_double_integrator.py A / B    (one part; summary_A/B.txt)
"""
import os
import time
from math import comb

import numpy as np
import casadi as ca
from scipy.linalg import solve_continuous_are, sqrtm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

U_MAX, V_MAX = 1.0, 2.0
M_GRID = 20                            # F4: grid points per sampling interval
Q_POS, Q_VEL, R_U = 10.0, 0.1, 0.1     # stage cost; terminal cost = stage cost without u


def reference(t):
    """x1 reference: 1.5 (beyond the wall) for 8 s, then -1 for 4 s, periodic."""
    return 1.5 if (t % 12.0) < 8.0 else -1.0


def zoh_step(x, u, T):
    """Exact ZOH map of the double integrator."""
    return np.array([x[0] + x[1] * T + 0.5 * u * T**2, x[1] + u * T])


def bernstein_coeffs(x1, x2, u, T):
    """beta_{c,nu}, nu = 0..3 (eq. (8)); power coefficients in s = t/T:
    a0 = c, a1 = -x2 T, a2 = -u T^2/2, a3 = -M_{c,3} T^3/6 = 0."""
    a = [1 - x1, -x2 * T, -0.5 * u * T**2, 0.0 * x1]
    return [sum(comb(n, j) / comb(3, j) * a[j] for j in range(n + 1)) for n in range(4)]


def grid_eps(T, m=M_GRID):
    """F4 tightening: max excess of x1(t) over its chord between grid points."""
    return U_MAX * (T / m)**2 / 8.0


def grid_constraints(x1, x2, u, T, m=M_GRID):
    """F4: c at the m+1 grid points of [0, T] minus the certified tightening."""
    eps = grid_eps(T, m)
    out = []
    for j in range(m + 1):
        t = j * T / m
        out.append(1 - (x1 + x2 * t + 0.5 * u * t**2) - eps)
    return out


def braking(x1, x2, lib=np):
    return x1 + (lib.fmax(x2, 0) if lib is ca else np.maximum(x2, 0))**2 / 2


# --------------------------------------------------------------------------
# Constructive terminal set (Theorem 1 + Corollary 1)
# --------------------------------------------------------------------------
def design_certified_terminal_set(Ts, R=8.7):
    """LQR ellipsoid S_L = {e' P e <= rho}, e = x - xe, with u = -K e.
    * |K e| <= 1 on S_L  (input admissibility)           -> rho
    * margin (10): L_F h >= -a h + delta with h = rho - e'Pe, u = -K e:
        L_F h = e' Qcl e >= mu e'Pe, Qcl = Q + K'RK, mu = lambda_min(P^-1/2 Qcl P^-1/2)
        => holds with a = mu, delta = mu * rho on all of S_L.
    * M_{h,2} >= |d^2 h/dt^2| on the reachable tube from S_L over [0,Ts]:
        h'' = -2 x'^T P x' - 2 e^T P x'',  x' = (e2, u), x'' = (u, 0)
        |h''| <= 2 lmax(P) (vK^2 + 1) + 2 lmax(P) eK,
        vK = vS + Ts, eK = eS + vS Ts + Ts^2/2 + Ts.
    * Theorem 1: T* = min{1/a, 2 delta / M_{h,2}} must be >= Ts.
    * Corollary 1: S_L subset of {c >= eta_c}, eta_c = M_{c,1} Ts + M_{c,2} Ts^2/2
      with M_{c,1} = vK, M_{c,2} = 1, M_{c,3} = 0  ->  xe1 = 1 - eta_c - e1S.
    """
    A = np.array([[0, 1.0], [0, 0]])
    B = np.array([[0], [1.0]])
    Q = np.eye(2)
    P = solve_continuous_are(A, B, Q, np.array([[R]]))
    K = (B.T @ P) / R
    Qcl = Q + R * K.T @ K
    Pih = np.linalg.inv(np.real(sqrtm(P)))
    mu = np.linalg.eigvalsh(Pih @ Qcl @ Pih).min()
    Pinv = np.linalg.inv(P)
    rho = 1.0 / float((K @ Pinv @ K.T).item())
    lmin, lmax = np.linalg.eigvalsh(P)[[0, -1]]
    eS, vS, e1S = np.sqrt(rho / lmin), np.sqrt(rho * Pinv[1, 1]), np.sqrt(rho * Pinv[0, 0])
    vK, eK = vS + Ts, eS + vS * Ts + Ts**2 / 2 + Ts
    Mh2 = 2 * lmax * (vK**2 + 1) + 2 * lmax * eK
    a, delta = mu, mu * rho
    Tstar = min(1 / a, 2 * delta / Mh2)
    eta_c = vK * Ts + Ts**2 / 2
    xe = np.array([1.0 - eta_c - e1S, 0.0])
    return dict(P=P, K=K, rho=rho, mu=mu, a=a, delta=delta, Mh2=Mh2, Tstar=Tstar,
                eta_c=eta_c, xe=xe, vS=vS, e1S=e1S, R=R)


def verify_certified_set(d, Ts, n=4000, seed=0):
    """Numerical sanity check of the certificate: from random points of S_L,
    the input u = -K e satisfies |u|<=1, the margin condition, beta_{h,nu}>=0,
    keeps h >= 0 between samples, and satisfies beta_{c,nu} >= 0."""
    rng = np.random.default_rng(seed)
    P, K, rho, xe = d["P"], d["K"], d["rho"], d["xe"]
    L = np.linalg.cholesky(np.linalg.inv(P) * rho)
    worst = dict(u=0, beta_h=np.inf, h_inter=np.inf, beta_c=np.inf)
    for _ in range(n):
        z = rng.normal(size=2)
        z = z / np.linalg.norm(z) * np.sqrt(rng.uniform()) ** 0.25    # bias to boundary
        e = L @ z
        x = xe + e
        u = float(-(K @ e).item())
        worst["u"] = max(worst["u"], abs(u))
        h = rho - e @ P @ e
        psi = -2 * e @ P @ np.array([x[1], u])
        betas_h = [h, h + Ts * psi / 2, h + Ts * psi - Ts**2 * d["Mh2"] / 2]
        worst["beta_h"] = min(worst["beta_h"], min(betas_h))
        for s in np.linspace(0, Ts, 21)[1:]:
            es = zoh_step(x, u, s) - xe
            worst["h_inter"] = min(worst["h_inter"], rho - es @ P @ es)
        worst["beta_c"] = min(worst["beta_c"], min(bernstein_coeffs(x[0], x[1], u, Ts)))
    return worst


# --------------------------------------------------------------------------
# MPC
# --------------------------------------------------------------------------
def build_mpc(method, N, T, gamma=0.3, with_cost=True, cert=None):
    opti = ca.Opti()
    X = opti.variable(2, N + 1)
    U = opti.variable(1, N)
    x0 = opti.parameter(2)
    ref = opti.parameter(N + 1)
    opti.subject_to(X[:, 0] == x0)
    cost = 0
    for i in range(N):
        xi, ui, xn = X[:, i], U[0, i], X[:, i + 1]
        opti.subject_to(xn[0] == xi[0] + xi[1] * T + 0.5 * ui * T**2)
        opti.subject_to(xn[1] == xi[1] + ui * T)
        opti.subject_to(opti.bounded(-U_MAX, ui, U_MAX))
        opti.subject_to(opti.bounded(-V_MAX, xn[1], V_MAX))
        cost += Q_POS * (xi[0] - ref[i])**2 + Q_VEL * xi[1]**2 + R_U * ui**2
        if method == "F1":
            opti.subject_to(1 - xn[0] >= (1 - gamma) * (1 - xi[0]))
        elif method == "F1h":
            hb_i = 1 - braking(xi[0], xi[1], ca)
            hb_n = 1 - braking(xn[0], xn[1], ca)
            opti.subject_to(hb_n >= (1 - gamma) * hb_i)
        elif method == "F2":
            if i >= 1:
                opti.subject_to(1 - xi[0] >= 0)
        elif method in ("F3", "F3C"):
            for b in bernstein_coeffs(xi[0], xi[1], ui, T):
                opti.subject_to(b >= 0)
        elif method == "F4":
            for gcon in grid_constraints(xi[0], xi[1], ui, T):
                opti.subject_to(gcon >= 0)
    xN = X[:, N]
    cost += Q_POS * (xN[0] - ref[N])**2 + Q_VEL * xN[1]**2          # terminal cost V_f
    if method == "F2":
        opti.subject_to(braking(xN[0], xN[1], ca) <= 1.0)
    elif method == "F3":
        opti.subject_to(braking(xN[0], xN[1], ca) <= 1.0 - T**2 / 18.0)
    elif method == "F3C":
        e = xN - ca.DM(cert["xe"])
        opti.subject_to(ca.mtimes([e.T, ca.DM(cert["P"]), e]) <= cert["rho"])
    elif method == "F4":
        opti.subject_to(braking(xN[0], xN[1], ca) <= 1.0 - grid_eps(T))
    opti.minimize(cost if with_cost else 0)
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
                          "ipopt.max_iter": 400, "ipopt.tol": 1e-8,
                          "ipopt.constr_viol_tol": 1e-8,
                          "ipopt.acceptable_constr_viol_tol": 1e-8})
    return opti, X, U, x0, ref, method, cert


def terminal_input(mpc, xN):
    """Input of Assumption 3 at the terminal state (last element of the shifted
    candidate in the proof of Theorem 2).  Braking sets (F2, F3, F4): u = -1 if
    x2 > 0 (keeps x1(t) below the braking level), u = 0 if x2 <= 0 (x1 decreases,
    and the node bound |x2| <= V_MAX is kept).  F3C: the certified LQR input
    u = -K (x - xe) of Theorem 1."""
    method, cert = mpc[5], mpc[6]
    if method == "F3C":
        return float(np.clip(-(cert["K"] @ (xN - cert["xe"])).item(), -U_MAX, U_MAX))
    return -U_MAX if xN[1] > 0 else 0.0


HAS_TERMINAL_SET = ("F2", "F3", "F3C", "F4")   # formulations covered by Theorem 2
CAND_TOL = 1e-7                                 # tolerance of the candidate check


def candidate_states(x, useq, T):
    """States x_0..x_N generated by the input sequence useq from x."""
    xs = [np.asarray(x, float)]
    for u in useq:
        xs.append(zoh_step(xs[-1], u, T))
    return np.array(xs).T


def candidate_ok(mpc, x, useq, T):
    """Check that the input sequence useq is feasible for the MPC of `mpc`
    from the state x (all constraints of build_mpc, tolerance CAND_TOL).
    Used before a shifted candidate is applied after a solver failure, so that
    a fallback step is only counted when a *verified* feasible point exists."""
    method, cert = mpc[5], mpc[6]
    xs = candidate_states(x, useq, T)
    tol = CAND_TOL
    if np.any(np.abs(useq) > U_MAX + tol) or np.any(np.abs(xs[1, 1:]) > V_MAX + tol):
        return False
    for i, u in enumerate(useq):
        x1, x2 = xs[0, i], xs[1, i]
        if method == "F2" and i >= 1 and 1 - x1 < -tol:
            return False
        if method in ("F3", "F3C") and min(bernstein_coeffs(x1, x2, u, T)) < -tol:
            return False
        if method == "F4" and min(grid_constraints(x1, x2, u, T)) < -tol:
            return False
    xN = xs[:, -1]
    if method == "F2":
        return braking(xN[0], xN[1]) <= 1.0 + tol
    if method == "F3":
        return braking(xN[0], xN[1]) <= 1.0 - T**2 / 18.0 + tol
    if method == "F4":
        return braking(xN[0], xN[1]) <= 1.0 - grid_eps(T) + tol
    if method == "F3C":
        e = xN - cert["xe"]
        return e @ cert["P"] @ e <= cert["rho"] + tol
    return False


def solve(mpc, x0_val, ref_val, warm=None):
    """Solve the MPC from x0_val.  warm = (U_guess, X_guess) or None.  Both
    initial guesses are always set explicitly (cold start: X = x0 at every
    node, U = 0), so that no guess leaks from a previous run or step; CasADi's
    Opti keeps the last set_initial values across solves."""
    opti, X, U, x0, ref = mpc[:5]
    N = U.shape[1]
    opti.set_value(x0, x0_val)
    opti.set_value(ref, ref_val)
    if warm is None:
        opti.set_initial(U, np.zeros((1, N)))
        opti.set_initial(X, np.tile(np.asarray(x0_val, float)[:, None], (1, N + 1)))
    else:
        opti.set_initial(U, np.reshape(warm[0], (1, N)))
        opti.set_initial(X, warm[1])
    try:
        sol = opti.solve()
        return True, np.atleast_1d(sol.value(U)), sol.value(X)
    except RuntimeError:
        return False, None, None


def closed_loop(mpc, T, N, t_final=24.0, x_init=(-2.0, 0.0), n_dense=50):
    """Closed loop with exact intersample evaluation.

    After a solver failure, the shifted candidate of the proof of Theorem 2
    (inputs u_1..u_{N-1} of the last solution followed by the terminal input
    of Assumption 3) is applied *if* candidate_ok() verifies that it satisfies
    all MPC constraints from the current state; this is a "fallback" step, and
    the remaining candidate is used as warm start of the next solve.  Only
    formulations with a terminal set (F2, F3, F3C, F4) have such a candidate.
    Otherwise the step counts as a loss of feasibility (n_inf) and full braking
    u = -1 is applied."""
    method = mpc[5]
    x = np.array(x_init, dtype=float)
    steps = int(round(t_final / T))
    dense, n_inf, ts, viol, cost, warm = [], 0, [], 0.0, 0.0, None
    t_near, n_fb, cand = None, 0, None
    for k in range(steps):
        t = k * T
        r = np.array([reference(t + i * T) for i in range(N + 1)])
        t0 = time.perf_counter()
        ok, useq, xseq = solve(mpc, x, r, warm)
        ts.append(time.perf_counter() - t0)
        if ok:
            u = float(useq[0])
            if method in HAS_TERMINAL_SET:
                # shifted candidate (proof of Thm. 2), also the next warm start
                cand = np.r_[useq[1:], terminal_input(mpc, xseq[:, -1])]
            else:
                cand = None                      # no terminal set: no candidate
            wu = cand if cand is not None else np.r_[useq[1:], useq[-1]]
            warm = (wu, candidate_states(zoh_step(x, u, T), wu, T))
        else:
            if cand is not None and candidate_ok(mpc, x, cand, T):
                # verified feasible point: suboptimal-MPC fallback
                n_fb += 1
                u = float(cand[0])
                xs_c = candidate_states(x, cand, T)
                cand = np.r_[cand[1:], terminal_input(mpc, xs_c[:, -1])]
                warm = (cand, candidate_states(zoh_step(x, u, T), cand, T))
            else:
                n_inf, u, warm, cand = n_inf + 1, -U_MAX, None, None
        for s in np.linspace(0.0, T, n_dense, endpoint=False):
            xs = zoh_step(x, u, s)
            dense.append((t + s, xs[0], xs[1], u))
            viol = max(viol, xs[0] - 1.0)
            if t_near is None and xs[0] >= 0.9:
                t_near = t + s
        cost += T * (Q_POS * (x[0] - reference(t))**2 + Q_VEL * x[1]**2 + R_U * u**2)
        x = zoh_step(x, u, T)
    return dict(dense=np.array(dense), n_inf=n_inf, viol=max(viol, 0.0), cost=cost,
                t_ms=1e3 * np.median(ts), t_near=t_near, n_fb=n_fb)


def monte_carlo(mpc, T, N, n_runs=300, t_final=10.0, seed=1):
    """Same random sample for every method; only starts feasible at t0 run
    (hypothesis of Theorem 2).  Every solve starts from the same cold initial
    guess, so the result does not depend on the order of the runs."""
    rng = np.random.default_rng(seed)
    X0 = np.column_stack([rng.uniform(-3.0, 1.0, n_runs), rng.uniform(-V_MAX, V_MAX, n_runs)])
    nf = nv = nl = nfb = 0
    worst = 0.0
    for x0 in X0:
        if not solve(mpc, x0, np.array([reference(i * T) for i in range(N + 1)]))[0]:
            continue
        nf += 1
        r = closed_loop(mpc, T, N, t_final=t_final, x_init=x0, n_dense=20)
        nv += r["viol"] > 1e-6
        nl += r["n_inf"] > 0
        nfb += r["n_fb"]
        worst = max(worst, r["viol"])
    return nf, nv, nl, worst, nfb


def feasible_boundary(method, T, N, x2_grid, gamma=0.3):
    mpc = build_mpc(method, N, T, gamma=gamma, with_cost=False)
    ref0, out = np.zeros(N + 1), []
    for x2 in x2_grid:
        lo, hi = -6.0, 1.0
        if not solve(mpc, np.array([lo, x2]), ref0)[0]:
            out.append(np.nan)
            continue
        if solve(mpc, np.array([hi, x2]), ref0)[0]:
            out.append(hi)
            continue
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if solve(mpc, np.array([mid, x2]), ref0)[0] else (lo, mid)
        out.append(lo)
    return np.array(out)


def _braking_boundary_check(T, xbar, coeffs, n=4001):
    """Assumption 3 on the boundary of S = {braking <= xbar, |x2| <= V_MAX}
    with the terminal input u = -1 (x2 > 0), u = 0 (x2 <= 0).  Returns
    (min of the intersample coefficients, max violation of F(x,u) in S)."""
    x2 = np.linspace(-V_MAX, V_MAX, n)
    x1 = xbar - np.maximum(x2, 0.0)**2 / 2
    u = np.where(x2 > 0, -U_MAX, 0.0)
    cmin = np.array(coeffs(x1, x2, u, T)).min()
    xn1, xn2 = x1 + x2 * T + 0.5 * u * T**2, x2 + u * T
    viol = max(np.max(braking(xn1, xn2) - xbar), np.max(np.abs(xn2) - V_MAX), 0.0)
    return cmin, viol


def check_terminal_compatibility(T, n=4001):
    """Assumption 3 for S_eta (xbar = 1 - T^2/18), Bernstein coefficients."""
    return _braking_boundary_check(T, 1.0 - T**2 / 18.0, bernstein_coeffs, n)


def check_terminal_compatibility_F4(T, n=4001):
    """Assumption 3 for the F4 terminal set (xbar = 1 - eps_m), gridded constraints."""
    return _braking_boundary_check(T, 1.0 - grid_eps(T), grid_constraints, n)


# --------------------------------------------------------------------------
class _Log(list):
    """List of summary lines that also prints each line immediately."""
    def append(self, line):
        print(line, flush=True)
        super().append(line)


def main(parts="AB"):
    L = _Log()
    if "A" in parts:
        part_a(L)
    if "B" in parts:
        part_b(L)
    name = "ex1_summary.txt" if parts == "AB" else f"ex1_summary_{parts}.txt"
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(L) + "\n")


def part_a(L):
    # ======================= (A) main setting ===========================
    T, N = 1.0, 8
    L.append(f"(A) Ts = {T}, N = {N}")
    cm, cv = check_terminal_compatibility(T)
    L.append(f"  Assumption 3 on boundary of S_eta (u=-1 if x2>0, else 0): min beta {cm:.2e}, "
             f"max violation of F(x,u) in S: {cv:.1e}")
    L.append("  Nominal run (24 s) and Monte Carlo (300 x0, 10 s):")
    L.append(f"  {'method':>8} {'viol':>7} {'#inf':>5} {'cost':>8} {'ms':>6} {'t(x1>=.9)':>9} |"
             f" {'#feas':>5} {'#viol':>5} {'#lost':>5} {'worst':>7}")
    cm, cv = check_terminal_compatibility_F4(T)
    L.append(f"  F4: M_GRID = {M_GRID}, eps_m = {grid_eps(T):.2e}; on boundary of its terminal set: "
             f"min gridded constraint {cm:.2e}, max violation of F(x,u) in S: {cv:.1e}")
    configs = [("F1", 0.3), ("F1h", 0.3), ("F2", None), ("F3", None), ("F4", None)]
    for m, g in configs:
        mpc = build_mpc(m, N, T, gamma=g or 0.3)
        r = closed_loop(mpc, T, N)
        np.savetxt(os.path.join(OUT, f"ex1_closed_loop_{m}.csv"), r["dense"][::2], delimiter=",",
                   header="t,x1,x2,u", comments="", fmt="%.5f")
        nf, nv, nl, w, nfb = monte_carlo(mpc, T, N)
        tn = f"{r['t_near']:.1f}" if r["t_near"] is not None else "-"
        L.append(f"  {m:>8} {r['viol']:7.4f} {r['n_inf']:5d} {r['cost']:8.2f} {r['t_ms']:6.2f} {tn:>9} |"
                 f" {nf:5d} {nv:5d} {nl:5d} {w:7.4f} fb={nfb}")
    L.append("  Gamma sweep of F1 (fairness of the baseline):")
    for g in (0.1, 0.3, 0.6, 1.0):
        mpc = build_mpc("F1", N, T, gamma=g)
        r = closed_loop(mpc, T, N)
        nf, nv, nl, w, nfb = monte_carlo(mpc, T, N)
        tn = f"{r['t_near']:.1f}" if r["t_near"] is not None else "-"
        L.append(f"  F1 g={g:<4} viol {r['viol']:.4f} cost {r['cost']:8.2f} t(x1>=.9) {tn:>5} |"
                 f" #feas {nf:4d} #viol {nv:4d} worst {w:.4f}")
    L.append("  Sampling-time sweep (horizon 8 s), nominal max violation:")
    for Ts in (0.25, 0.5, 1.0):
        row = [f"Ts={Ts:4.2f}"]
        for m in ("F1", "F1h", "F2", "F3", "F4"):
            row.append(f"{m}:{closed_loop(build_mpc(m, int(round(8 / Ts)), Ts), Ts, int(round(8 / Ts)))['viol']:.4f}")
        L.append("  " + "  ".join(row))
    x2g = np.linspace(-2.0, 2.0, 41)
    bnd = {m: feasible_boundary(m, T, N, x2g) for m in ("F1", "F1h", "F2", "F3", "F4")}
    kernel = 1.0 - np.maximum(x2g, 0.0)**2 / 2
    np.savetxt(os.path.join(OUT, "ex1_feasible_boundary.csv"),
               np.column_stack([x2g, kernel] + [bnd[m] for m in ("F1", "F1h", "F2", "F3", "F4")]),
               delimiter=",", header="x2,kernel,F1,F1h,F2,F3,F4", comments="", fmt="%.5f")
    for m in bnd:
        g = bnd[m] - kernel
        L.append(f"  boundary {m}: max excess over kernel {np.nanmax(g):+.4f}, "
                 f"min {np.nanmin(g):+.4f}, mean {np.nanmean(g):+.4f}")



def part_b(L):
    # ================= (B) constructive pipeline ========================
    T, N = 0.1, 80
    d = design_certified_terminal_set(T)
    L.append(f"\n(B) Constructive route, Ts = {T}, N = {N}")
    L.append(f"  LQR R={d['R']}, rho={d['rho']:.3f}, a=mu={d['mu']:.4f}, delta={d['delta']:.4f}, "
             f"M_h2={d['Mh2']:.2f}, T*={d['Tstar']:.4f} (>= Ts: {d['Tstar'] >= T}), "
             f"eta_c={d['eta_c']:.4f}, xe1={d['xe'][0]:.4f}, max x1 on S_L={d['xe'][0] + d['e1S']:.4f}")
    w = verify_certified_set(d, T)
    L.append(f"  certificate check: max|u|={w['u']:.4f}, min beta_h={w['beta_h']:.3e}, "
             f"min h between samples={w['h_inter']:.3e}, min beta_c={w['beta_c']:.3e}")
    L.append(f"  {'method':>8} {'viol':>9} {'#inf':>5} {'cost':>8} {'ms':>6} |"
             f" {'#feas':>5} {'#viol':>5} {'#lost':>5} {'worst':>9}")
    cm, cv = check_terminal_compatibility_F4(T)
    L.append(f"  F4: eps_m = {grid_eps(T):.2e}, on boundary of its terminal set: min gridded "
             f"constraint {cm:.2e}, max violation of F(x,u) in S: {cv:.1e}")
    for m in ("F2", "F3", "F3C", "F4"):
        mpc = build_mpc(m, N, T, cert=d)
        r = closed_loop(mpc, T, N)
        np.savetxt(os.path.join(OUT, f"ex1B_closed_loop_{m}.csv"), r["dense"][::10], delimiter=",",
                   header="t,x1,x2,u", comments="", fmt="%.5f")
        nf, nv, nl, wv, nfb = monte_carlo(mpc, T, N, n_runs=100)
        L.append(f"  {m:>8} {r['viol']:9.6f} {r['n_inf']:5d} {r['cost']:8.2f} {r['t_ms']:6.2f} |"
                 f" {nf:5d} {nv:5d} {nl:5d} {wv:9.6f} fb={nfb}")


if __name__ == "__main__":
    # "python3 ex1_double_integrator.py" runs everything; "A" or "B" runs one
    # part (useful to run both parts in parallel), see README.
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "AB")
