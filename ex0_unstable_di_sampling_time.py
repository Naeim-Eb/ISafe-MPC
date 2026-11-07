#!/usr/bin/env python3
"""
Example 1 of the paper (unstable double integrator): a set certified by a valid
continuous-time CBF that is not discrete-time controlled invariant under ZOH,
and validation of Theorem 1 (explicit sampling-time bound).

System:   x1' = x2,  x2' = 2 x2 + u,  |u| <= 1.
Set:      S = { h(x) = rho - 4 x1^2 - (x1 + x2)^2 >= 0 },  rho = 0.01.
Feedback: u = -5 x1 - 4 x2 gives  h' = -2 h + 2 rho  (valid CBF, margin a = 2,
          delta = 2 rho on all of S).

The script checks every number stated in the paper:
 (a) |u| <= 0.41 on S for the feedback above, and h' = -2h + 2rho (symbolically).
 (b) Ts = 1.5 s, x = (0, 0.1) on the boundary of S: max_{|u|<=1} h(x_{k+1}) < 0,
     and min over ALL u in R of 4x1^2 + (x1+x2)^2 at x_{k+1} exceeds rho,
     so S is not DT-CI.
 (c) The largest Ts for which S is still DT-CI (scan over the boundary).
 (d) Bounds on S (|x1|, |x2|), the tube box R = {|x1|<=0.06, |x2|<=0.18}
     for Ts <= 0.02 s, M_{h,2} = sup_{R x U} |h''|, and T* of eq. (Tstar).
 (e) Closed loop at Ts = T*: from random initial states in S (half of them on
     the boundary), at every step the input in U that MINIMISES h' subject to
     the Taylor-Bernstein constraints beta_{h,nu} >= 0 (q = 1) is applied
     (an adversarial choice among the inputs that Theorem 1 admits). h is
     evaluated densely between samples with the exact (matrix-exponential)
     solution.

Requirements: numpy, scipy, sympy.     Run: python3 ex0_unstable_di_sampling_time.py
"""
import os
import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

RHO = 0.01
A_RATE, DELTA = 2.0, 2 * RHO
A = np.array([[0.0, 1.0], [0.0, 2.0]])
B = np.array([0.0, 1.0])
log = []


def say(s):
    print(s)
    log.append(s)


def h(x1, x2):
    return RHO - 4 * x1**2 - (x1 + x2)**2


def zoh(Ts):
    """Exact ZOH map x+ = Ad x + Bd u."""
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2] = B
    E = expm(M * Ts)
    return E[:2, :2], E[:2, 2]


def boundary(n):
    """Parametrisation of the boundary of S: 2 x1 = 0.1 cos a, x1 + x2 = 0.1 sin a."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x1 = 0.5 * np.sqrt(RHO) * np.cos(a)
    x2 = np.sqrt(RHO) * np.sin(a) - x1
    return x1, x2


# ------------------------------------------------------------------ (a)
X1, X2, U = sp.symbols("x1 x2 u", real=True)
hs = RHO - 4 * X1**2 - (X1 + X2)**2
hdot = sp.diff(hs, X1) * X2 + sp.diff(hs, X2) * (2 * X2 + U)
check = sp.simplify(hdot.subs(U, -5 * X1 - 4 * X2) - (-2 * hs + 2 * RHO))
say(f"(a) h' + 2h - 2rho under u = -5x1-4x2 (should be 0): {check}")
# max |u| on S: maximise |-z - 4y| s.t. 4 z^2 + y^2 <= rho  (z = x1, y = x1 + x2)
umax_S = np.sqrt(RHO) * np.sqrt(1 / 4 + 16)
say(f"    max |u| on S = {umax_S:.4f}  (paper: <= 0.41)")

# ------------------------------------------------------------------ (b)
Ts = 1.5
Ad, Bd = zoh(Ts)
x = np.array([0.0, 0.1])
say(f"(b) Ts = {Ts}: h(0, 0.1) = {h(*x):.2e}")
us = np.linspace(-1, 1, 200001)
xn = Ad @ x[:, None] + Bd[:, None] * us[None, :]
say(f"    max_|u|<=1 h(x_k+1) = {h(xn[0], xn[1]).max():.5f}  (paper: < 0)")
# minimum over all real u of q(u) = 4 x1^2 + (x1+x2)^2 along the line x0 + Bd u
Q = np.array([[5.0, 1.0], [1.0, 1.0]])            # 4x1^2 + (x1+x2)^2 = x^T Q x
p0 = Ad @ x
ustar = -(Bd @ Q @ p0) / (Bd @ Q @ Bd)
pmin = p0 + Bd * ustar
say(f"    min over all u in R of 4x1^2+(x1+x2)^2 = {pmin @ Q @ pmin:.5f} at u = {ustar:.3f}"
    f"  (> rho = {RHO}: no input keeps x_k+1 in S)")
say(f"    x2 amplification over one hold: e^(2Ts) = {np.exp(2*Ts):.2f}")

# ------------------------------------------------------------------ (c)
bx1, bx2 = boundary(1440)
us_c = np.linspace(-1, 1, 2001)


def worst_boundary(Ts):
    Ad, Bd = zoh(Ts)
    P = Ad @ np.vstack([bx1, bx2])
    worst = np.inf
    for k in range(P.shape[1]):
        xn = P[:, k:k + 1] + Bd[:, None] * us_c[None, :]
        worst = min(worst, h(xn[0], xn[1]).max())
    return worst


say("(c) min over boundary of max_|u|<=1 h(x_k+1):")
for T in [0.5, 1.0, 1.1, 1.15, 1.2, 1.25, 1.5]:
    say(f"    Ts = {T:5.2f}: {worst_boundary(T):+.5f}")
lo, hi = 1.0, 1.5
for _ in range(25):
    mid = 0.5 * (lo + hi)
    if worst_boundary(mid) >= 0:
        lo = mid
    else:
        hi = mid
say(f"    boundary of S can be kept in S up to Ts ~= {lo:.3f} s")


def worst_all(Ts):
    """Same check over the whole of S (41 radii x 721 angles)."""
    Ad, Bd = zoh(Ts)
    worst = np.inf
    for r in np.linspace(0, 1, 41):
        P = Ad @ np.vstack([r * bx1[::2], r * bx2[::2]])
        for k in range(P.shape[1]):
            xn = P[:, k:k + 1] + Bd[:, None] * us_c[None, :]
            worst = min(worst, h(xn[0], xn[1]).max())
    return worst


for T in [1.1, 1.15]:
    say(f"    whole S, Ts = {T}: min_x max_u h(x_k+1) = {worst_all(T):+.5f}"
        f"  ({'DT-CI' if worst_all(T) >= 0 else 'not DT-CI'})")

# ------------------------------------------------------------------ (d)
x1max = 0.5 * np.sqrt(RHO)
x2max = np.sqrt(RHO) * np.sqrt(1 / 4 + 1)
say(f"(d) on S: |x1| <= {x1max:.4f}, |x2| <= {x2max:.4f}")
TsR = 0.02
say(f"    Ts <= {TsR}: |x1| <= {x1max + TsR*0.18:.4f} <= 0.06, "
    f"|x2| <= {x2max + TsR*(2*0.18+1):.4f} <= 0.18 on the tube (bootstrap with R)")


def hdd(v):
    x1, x2, u = v
    return -8 * (x2**2 + x1 * (2 * x2 + u)) - 2 * ((3 * x2 + u)**2 + 3 * (x1 + x2) * (2 * x2 + u))


g = np.linspace(-1, 1, 121)
G1, G2, GU = np.meshgrid(0.06 * g, 0.18 * g, g, indexing="ij")
Mgrid = np.abs(hdd((G1, G2, GU))).max()
best = 0.0
bnds = [(-0.06, 0.06), (-0.18, 0.18), (-1, 1)]
rng = np.random.default_rng(0)
for _ in range(200):
    v0 = [rng.uniform(*b) for b in bnds]
    for sgn in (1, -1):
        r = minimize(lambda v: -sgn * hdd(v), v0, bounds=bnds)
        best = max(best, abs(hdd(r.x)))
M_h2 = max(Mgrid, best)
say(f"    M_h2 = sup_RxU |h''| = {M_h2:.4f}  (grid {Mgrid:.4f}, local opt {best:.4f}); paper uses 7.62")
M_h2 = 7.62
Tstar = min(1 / A_RATE, 2 * DELTA / M_h2)
say(f"    T* = min(1/a, 2 delta/M_h2) = min({1/A_RATE}, {2*DELTA/M_h2:.5f}) = {Tstar*1e3:.3f} ms")

# ------------------------------------------------------------------ (e)
Ts = Tstar
Ad, Bd = zoh(Ts)
nsub = 50
taus = np.linspace(0, Ts, nsub + 1)[1:]
sub = [zoh(t) for t in taus]


def grad_h(x):
    x1, x2 = x
    return np.array([-8 * x1 - 2 * (x1 + x2), -2 * (x1 + x2)])


def adversarial_input(x):
    """u in [-1,1] minimising psi(u) = L_f h + L_g h u subject to
    beta_1 = h + Ts/2 psi >= 0 and beta_2 = h + Ts psi - Ts^2/2 M_h2 >= 0."""
    gh = grad_h(x)
    Lf = gh @ (A @ x)
    Lg = gh @ B
    hx = h(*x)
    psi_min = max(-2 * hx / Ts, -hx / Ts + Ts * M_h2 / 2)   # lower bound on psi
    # feasible u: Lf + Lg u >= psi_min,  u in [-1, 1]
    if abs(Lg) < 1e-14:
        feas = Lf >= psi_min - 1e-14
        return (1.0 if Lf >= 0 else -1.0), feas
    ub = (psi_min - Lf) / Lg
    if Lg > 0:
        lo_, hi_ = max(-1.0, ub), 1.0
        u = lo_                      # smallest psi
    else:
        lo_, hi_ = -1.0, min(1.0, ub)
        u = hi_
    return u, lo_ <= hi_ + 1e-14


n_runs, T_end = 300, 2.0
steps = int(round(T_end / Ts))
min_h, n_infeas = np.inf, 0
for k in range(n_runs):
    if k % 2 == 0:                   # on the boundary
        a = rng.uniform(0, 2 * np.pi)
        r = 1.0
    else:                            # in the interior
        a = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(rng.uniform())
    x1 = r * 0.5 * np.sqrt(RHO) * np.cos(a)
    x = np.array([x1, r * np.sqrt(RHO) * np.sin(a) - x1])
    for _ in range(steps):
        u, ok = adversarial_input(x)
        n_infeas += (not ok)
        for (Adt, Bdt) in sub:
            xt = Adt @ x + Bdt * u
            min_h = min(min_h, h(*xt))
        x = Ad @ x + Bd * u
say(f"(e) Ts = T* = {Ts*1e3:.3f} ms, {n_runs} runs x {steps} steps, {nsub} points/interval,"
    f" adversarial admissible input:")
say(f"    steps with no admissible input: {n_infeas};  min_t h(x(t)) = {min_h:.3e}  (>= 0 required)")

with open(os.path.join(OUT, "ex0_summary.txt"), "w") as fh:
    fh.write("\n".join(log) + "\n")
