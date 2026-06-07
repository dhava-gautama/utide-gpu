"""
Prototype a mixed-precision GPU basis: astronomical reduction (huge day
numbers) stays FP64; the expensive complex matmul + transcendentals run in
FP32. Compare accuracy vs the FP64 basis and measure speed.
"""
import sys; sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import numpy as np, time
import cupy as cp
from utide.harmonics import ut_E                       # FP64 CPU reference
from utide._harmonics_xp import _Tables, ut_astron_xp, get_tables, _ishallow, _nshallow, _kshallow
from utide._ut_constants import ut_constants
shallow = ut_constants.shallow

def sync(): cp.cuda.Stream.null.synchronize()
def bench(fn, n=5):
    fn(); sync(); best = 1e18
    for _ in range(n):
        t0 = time.perf_counter(); fn(); sync(); best = min(best, time.perf_counter()-t0)
    return best

tab = get_tables(cp)                                   # FP64 tables
# FP32 copies of the matmul/elementwise tables
satsel32 = tab.satsel.astype(cp.float32)

def FUV_mixed(t, lind, lat, final_fp32_phase):
    nt = len(t)
    astro = ut_astron_xp(cp, t, tab)                   # FP64 (accurate)
    if abs(lat) < 5: lat = np.sign(lat)*5
    slat = np.sin(np.deg2rad(lat))
    rr = tab.amprat.copy()
    rr = cp.where(tab.ilatfac==1, rr*0.36309*(1-5*slat**2)/slat, rr)
    rr = cp.where(tab.ilatfac==2, rr*2.59808*slat, rr)
    uu = tab.deldood @ astro[3:6,:] + tab.phcorr[:,None]   # FP64 reduction
    uu = cp.fmod(uu, 1)
    mat = (rr[:,None]*cp.exp(1j*2*np.pi*uu)).astype(cp.complex64)   # -> FP32
    F = (1.0 + satsel32 @ mat)                          # FP32 matmul (the 40x win)
    U = cp.angle(F)/(2*np.pi); F = cp.abs(F)            # FP32
    for k, j, coef, acoef in tab._shallow:
        F[k,:] = cp.prod(F[j,:]**acoef.astype(cp.float32), axis=0)
        U[k,:] = cp.sum(U[j,:]*coef.astype(cp.float32), axis=0)
    lind_x = cp.asarray(lind)
    F = F[lind_x,:].T; U = U[lind_x,:].T                # FP32
    # V (Greenwich phase) kept FP64 for accuracy
    astro = ut_astron_xp(cp, t, tab)
    V = tab.doodson @ astro + tab.semi[:,None]; V = cp.fmod(V,1)
    for k, j, coef, _a in tab._shallow:
        V[k,:] = cp.sum(V[j,:]*coef, axis=0)
    V = V[lind_x,:].T                                   # FP64
    # final assembly
    if final_fp32_phase:
        phase = (U + V.astype(cp.float32))
        E = (F * cp.exp(1j*2*np.pi*phase)).astype(cp.complex64)
    else:
        phase = U.astype(cp.float64) + V               # FP64 phase
        E = (F.astype(cp.float64) * cp.exp(1j*2*np.pi*phase)).astype(cp.complex64)
    return E

# constituent set
import utide
valid=set(n.strip() for n in ut_constants.const.name)
want=['M2','S2','N2','K2','K1','O1','P1','Q1','M4','MS4','M6','MM','MF','SA','MU2','J1','OO1']
constit=[c for c in want if c in valid]
nt0=8760; t0=np.arange(nt0)/24.0
c0=utide.solve(t0, np.cos(2*np.pi*t0)+0.1*np.random.randn(nt0), lat=45, constit=constit, method="ols", conf_int="none", verbose=False)
lind=c0['aux']['lind']; frq=c0['aux']['frq']; lat=45.0

print("ACCURACY vs FP64 CPU ut_E (max abs error on basis E):")
t=np.arange(20000)/24.0; tref=t.mean()
Eref=ut_E(t,tref,frq,lind,lat,[0,0,0,0],[])
for tag,f32p in [("FP64 phase",False),("FP32 phase",True)]:
    Em=cp.asnumpy(FUV_mixed(cp.asarray(t),lind,lat,f32p))
    # convert phase error to degrees: error in E ~ amplitude*angle_err
    err=np.abs(Em-Eref).max()
    print(f"  mixed ({tag:10s}): max|ΔE| = {err:.2e}")

print("\nSPEED (basis build on GPU):")
from utide._harmonics_xp import ut_E_xp
print(f"{'nt':>8} {'FP64(ms)':>9} {'mixed-FP32(ms)':>14} {'speedup':>8}")
for nd in [10*365, 30*365]:
    nt=nd*24; t=np.arange(nt)/24.0; tref=t.mean(); td=cp.asarray(t)
    t64=bench(lambda: ut_E_xp(cp, td, tref, frq, lind, lat, [0,0,0,0]))
    tmx=bench(lambda: FUV_mixed(td, lind, lat, True))
    print(f"{nt:>8} {1000*t64:>9.1f} {1000*tmx:>14.1f} {t64/tmx:>7.1f}x")
