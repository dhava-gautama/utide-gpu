import sys; sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import numpy as np, warnings, time
warnings.filterwarnings("ignore")
import utide

path = "/home/dhava/utide-gpu/UTide/notebooks/can1998.dtf"
raw = np.loadtxt(path)                       # seconds year month day hour elev flag
seconds, elev, flag = raw[:, 0], raw[:, 5].copy(), raw[:, 6]
elev[flag == 2] = np.nan                     # bad points
elev[np.abs(elev - 9.990) < 1e-6] = np.nan   # missing sentinel
t = seconds / 86400.0                        # days since 1998-01-01
anom = elev - np.nanmean(elev)

kw = dict(lat=-25, method="ols", conf_int="linear", epoch="1998-01-01", verbose=False)
print("Real tidal record (can1998, hourly, 1 yr), auto constituents")
c0 = utide.solve(t, anom, **kw)
c1 = utide.solve(t, anom, gpu=True, **kw)
print(f"  nconstit = {len(c0['name'])}\n")

# align by name (order may match, but be safe)
names0 = list(c0['name']); names1 = list(c1['name'])
order = [names1.index(n) for n in names0]
print(f"{'field':8s} {'max|Δ|':>11s} {'max rel':>11s}   verdict")
allok = True
for k in ["A", "g", "A_ci", "g_ci", "mean"]:
    a = np.atleast_1d(np.asarray(c0[k], float))
    b = np.atleast_1d(np.asarray(c1[k], float))
    if b.shape == a.shape and k in ("A", "g", "A_ci", "g_ci"):
        b = b[order]
    fin = np.isfinite(a) & np.isfinite(b)
    d = np.max(np.abs(a[fin] - b[fin])) if fin.any() else 0.0
    rel = np.max(np.abs(a[fin] - b[fin]) / (np.abs(a[fin]) + 1e-9)) if fin.any() else 0.0
    ok = rel < 1e-5 or d < 1e-6
    allok &= ok
    print(f"{k:8s} {d:11.2e} {rel:11.2e}   {'OK' if ok else 'CHECK'}")

# show the top constituents both ways
print("\nTop-8 constituents (amp / phase):  CPU   |   GPU")
idx = np.argsort(c0['A'])[::-1][:8]
for i in idx:
    j = names1.index(names0[i])
    print(f"  {names0[i]:5s}  A={c0['A'][i]:7.4f}/{c1['A'][j]:7.4f}  g={c0['g'][i]:7.2f}/{c1['g'][j]:7.2f}")

print(f"\nVERDICT: {'PASS - GPU matches CPU' if allok else 'CHECK differences above'}")
