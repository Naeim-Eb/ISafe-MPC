#!/usr/bin/env python3
"""Median solve times of the nominal closed-loop runs (Table I, column
t_solve), measured sequentially in one process with nothing else running, so
that all methods are timed under the same conditions (the experiment scripts
may be run in parallel, which inflates their own timings).
Run after the experiment scripts:  python3 timing.py"""
import os
import ex1_double_integrator as e1
import ex2_pvtol as e2

if __name__ == "__main__":
    lines = []
    for m in ("F1", "F1h", "F2", "F3", "F4"):
        r = e1.closed_loop(e1.build_mpc(m, 8, 1.0), 1.0, 8)
        lines.append(f"Ex1  (Ts=1)   {m:>4}: median solve {r['t_ms']:7.2f} ms, cost {r['cost']:.2f}")
    d = e1.design_certified_terminal_set(0.1)
    for m in ("F2", "F3", "F3C", "F4"):
        r = e1.closed_loop(e1.build_mpc(m, 80, 0.1, cert=d), 0.1, 80)
        lines.append(f"Ex1B (Ts=0.1) {m:>4}: median solve {r['t_ms']:7.2f} ms, cost {r['cost']:.2f}")
    for m in e2.METHODS:
        r = e2.closed_loop(m, 0.1, 20)
        lines.append(f"Ex2  (Ts=0.1) {m:>4}: median solve {r['t_solve_ms']:7.2f} ms, cost {r['cost']:.2f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(e1.OUT, "timing_summary.txt"), "w") as f:
        f.write(txt + "\n")
