import sys

sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from utide import solve
from utide._harmonics_xp import ut_E_xp

rng = np.random.default_rng(11)
nt = 2 * 365 * 24
t = np.arange(nt) / 24.0
fr = {"M2": 1 / 12.42, "S2": 1 / 12.0, "K1": 1 / 23.93, "O1": 1 / 25.82}
u = (
    np.cos(2 * np.pi * fr["M2"] * 24 * t + 0.3)
    + 0.4 * np.cos(2 * np.pi * fr["S2"] * 24 * t + 1.0)
    + 0.5 * np.cos(2 * np.pi * fr["K1"] * 24 * t + 2.0)
    + 0.1 * rng.standard_normal(nt)
)
sets = [
    ["M2", "S2", "N2", "K1", "O1", "P1", "Q1", "M4", "M6", "MM", "MF"],
    ["M2", "S2", "N2", "K1", "O1", "Q1", "M4"],
    ["M2", "S2", "N2", "K2", "K1", "O1"],
    ["M2", "S2", "N2", "K1", "O1", "M4", "M6"],
]
for C in sets:
    c = solve(t, u, lat=45, constit=C, method="ols", conf_int="none", verbose=False)
    lind = c["aux"]["lind"]
    frq = c["aux"]["frq"]
    E = ut_E_xp(np, t, t.mean(), frq, lind, 45, [0, 0, 0, 0])
    B = np.hstack((E, E.conj(), np.ones((nt, 1)), (t - t.mean())[:, None]))
    print(
        f"n={len(C):2d} cond(B)={np.linalg.cond(B):.2e}  maxA={np.max(c['A']):.4f}  set={C}",
    )
