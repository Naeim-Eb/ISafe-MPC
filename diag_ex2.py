"""Diagnostic: for each MPC failure, report IPOPT status and the constraint
violation of the shifted candidate used in the proof of Theorem 2."""
import sys, numpy as np, casadi as ca
import ex2_pvtol as e

def diagnose(method, Ts, N, x0, t_final=12.0):
    mpc = e.build_mpc(method, N, Ts); o = mpc["opti"]; Fd = e.rk4_zoh(Ts)
    x = np.array(x0, float); Xw = Uw = None; prev = None
    for k in range(int(round(t_final / Ts))):
        o.set_value(mpc["x0"], x)
        o.set_value(mpc["ref"], np.column_stack([e.waypoint(k * Ts + i * Ts) for i in range(N + 1)]))
        if Xw is not None:
            o.set_initial(mpc["X"], Xw); o.set_initial(mpc["U"], Uw)
        else:
            o.set_initial(mpc["X"], np.tile(x[:, None], (1, N + 1)))
            o.set_initial(mpc["U"], np.tile([[e.M * e.G], [0.0]], (1, N)))
        try:
            sol = o.solve(); U = sol.value(mpc["U"]); X = sol.value(mpc["X"])
        except RuntimeError:
            st = o.stats()["return_status"]
            if prev is None:
                return dict(k=k, status=st, cand=None)
            Up = prev
            Uc = np.column_stack([Up[:, 1:], [e.M * e.G, 0.0]])
            xc = x.copy(); worst = np.inf
            bw = e.bernstein_wall_sd if method == "F3SD" else (lambda a, b, T: e.bernstein_wall(a, b, T, e.remainder_bound(T)))
            bf = e.bernstein_floor_sd if method == "F3SD" else (lambda a, b, T: e.bernstein_floor(a, b, T, e.remainder_bound(T)))
            for i in range(N):
                b = [float(v) for v in bw(ca.DM(xc), ca.DM(Uc[:, i]), Ts) + bf(ca.DM(xc), ca.DM(Uc[:, i]), Ts)]
                worst = min(worst, min(b))
                xc = np.array(Fd(xc, Uc[:, i])).ravel()
            eta = e.eta_for(method, Ts)
            term = min(e.Y_WALL - xc[0] - eta, xc[1] - e.Z_FLOOR - eta, -np.abs(xc[2:]).max())
            return dict(k=k, status=st, cand_min_beta=worst, cand_terminal=term, x=np.round(x, 4))
        prev = U
        Xw = np.column_stack([X[:, 1:], X[:, -1]]); Uw = np.column_stack([U[:, 1:], [e.M * e.G, 0.0]])
        x, _, _ = e.plant_interval(x, U[:, 0], Ts, 3)
    return "no failure"

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    X0 = [np.array([rng.uniform(0.0, 1.8), rng.uniform(0.2, 2.0), 0.0, rng.uniform(-1.0, 2.0), rng.uniform(-2.0, 1.0), 0.0]) for _ in range(60)]
    which = sys.argv[1]
    if which == "mc":
        for m in ("F3", "F3SD"):
            mpc = e.build_mpc(m, 20, 0.1)
            for j, x0 in enumerate(X0):
                if not e.initially_feasible(mpc, x0, 0.1):
                    continue
                r = e.closed_loop(m, 0.1, 20, t_final=12.0, x_init=x0, n_dense=3, mpc=mpc)
                if r["n_infeas"] > 0:
                    print(m, "run", j, "x0", np.round(x0, 3), "first infeas step", r["first_infeas"], flush=True)
                    print("   ", diagnose(m, 0.1, 20, x0), flush=True)
    else:
        print("F3SD Ts=0.2:", diagnose("F3SD", 0.2, 10, [0, 1, 0, 0, 0, 0]), flush=True)
