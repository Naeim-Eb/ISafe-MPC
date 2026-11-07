#!/usr/bin/env python3
"""
Converts the CSV files produced by ex1_double_integrator.py and ex2_pvtol.py
into down-sampled pgfplots coordinate lists, written to
./results/figure_coords.txt.  The coordinate lists are pasted verbatim into
the \\addplot coordinates {...} commands of Figs. 4 and 5 of the paper, so
the LaTeX source stays self-contained and editable.

Run after the two example scripts:   python3 make_figure_data.py
"""
import os
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def coords(xs, ys, fmt="{:.3f}"):
    return " ".join("(" + fmt.format(x) + "," + fmt.format(y) + ")"
                    for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y))


def main():
    out = []
    # ---- Fig. 3(a): Example 1, x1(t) in the wall-contact window t in [1.9, 7.7]
    for m in ("F1", "F2", "F3"):
        d = np.genfromtxt(os.path.join(RES, f"ex1_closed_loop_{m}.csv"), delimiter=",", names=True)
        k = (d["t"] >= 1.9 - 1e-9) & (d["t"] <= 7.7 + 1e-9)
        out.append(f"% ex1 x1(t) {m}\n" + coords(d["t"][k], d["x1"][k]))
    # ---- Fig. 3(b): Example 1, feasible-set boundaries
    b = np.genfromtxt(os.path.join(RES, "ex1_feasible_boundary.csv"), delimiter=",", names=True)
    for m in ("kernel", "F1", "F1h", "F2", "F3", "F4"):
        out.append(f"% ex1 boundary {m}\n" + coords(b["x2"], b[m]))
    # ---- Fig. 4(a): Example 2, (y, z) path at t = 0, 0.04, 0.08, ... s, taken
    #      from the full-resolution trajectory (grid 0.004 s)
    for m in ("F1", "F2", "F3", "F3SD"):
        d = np.genfromtxt(os.path.join(RES, f"ex2_closed_loop_{m}.csv"), delimiter=",", names=True)
        k = np.abs(d["t"] / 0.04 - np.round(d["t"] / 0.04)) < 1e-6
        out.append(f"% ex2 path {m}\n" + coords(d["y"][k], d["z"][k]))
    # ---- Fig. 4(b): Example 2, minimum of min(c_wall, c_floor) over each
    #      sampling interval (computed in ex2_pvtol.py from the full-resolution
    #      trajectory), in millimetres.  The const-plot list is closed with the
    #      end point t = 12 s; intervals with a violation (> 1e-3 mm) are listed separately
    #      (markers at the interval centres).
    for m in ("F1", "F2", "F3", "F3SD"):
        d = np.genfromtxt(os.path.join(RES, f"ex2_interval_min_{m}.csv"), delimiter=",", names=True)
        ts, mins = d["t_k"], d["min_margin_mm"]
        out.append(f"% ex2 interval-min margin [mm] {m}\n"
                   + coords(np.r_[ts, ts[-1] + 0.1], np.r_[mins, mins[-1]], "{:.2f}"))
        # same threshold as the violation count of Table I (1e-6 m = 1e-3 mm)
        neg = mins < -1e-3
        if neg.any():
            out.append(f"% ex2 violating intervals (centre, margin [mm]) {m}\n"
                       + coords(ts[neg] + 0.05, mins[neg], "{:.3f}"))
    with open(os.path.join(RES, "figure_coords.txt"), "w") as f:
        f.write("\n\n".join(out) + "\n")
    print("wrote", os.path.join(RES, "figure_coords.txt"))


if __name__ == "__main__":
    main()
