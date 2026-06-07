import sys; sys.path.insert(0, "/home/dhava/utide-gpu/UTide")
import numpy as np, time
import cupy as cp

def sync(): cp.cuda.Stream.null.synchronize()
def bench(fn, n=10):
    fn(); sync()
    best = 1e18
    for _ in range(n):
        t0 = time.perf_counter(); fn(); sync()
        best = min(best, time.perf_counter() - t0)
    return best

print("Ceiling check: FP32 vs FP64 for the dominant basis ops (RTX 4060)")
print(f"{'op':28s} {'nt':>8} {'FP64(ms)':>9} {'FP32(ms)':>9} {'speedup':>8}")
for nd in [10*365, 30*365]:
    nt = nd*24
    # 1) complex matmul like satsel @ mat : (146,162)@(162,nt)
    A64 = cp.asarray(np.random.randn(146,162)); m64 = (cp.random.standard_normal((162,nt)) + 1j*cp.random.standard_normal((162,nt)))
    A32 = A64.astype(cp.float32); m32 = m64.astype(cp.complex64)
    t64 = bench(lambda: A64 @ m64); t32 = bench(lambda: A32 @ m32)
    print(f"{'matmul (146,162)@(162,nt)':28s} {nt:>8} {1000*t64:>9.2f} {1000*t32:>9.2f} {t64/t32:>7.1f}x")
    # 2) complex exp over (162, nt)
    u64 = cp.random.standard_normal((162,nt)); u32 = u64.astype(cp.float32)
    t64 = bench(lambda: cp.exp(1j*2*np.pi*u64)); t32 = bench(lambda: cp.exp(1j*2*np.pi*u32.astype(cp.complex64)))
    print(f"{'exp(1j*2pi*u) (162,nt)':28s} {nt:>8} {1000*t64:>9.2f} {1000*t32:>9.2f} {t64/t32:>7.1f}x")
    # 3) angle + abs over (146, nt) complex
    F64 = cp.random.standard_normal((146,nt)) + 1j*cp.random.standard_normal((146,nt)); F32 = F64.astype(cp.complex64)
    t64 = bench(lambda: (cp.angle(F64), cp.abs(F64))); t32 = bench(lambda: (cp.angle(F32), cp.abs(F32)))
    print(f"{'angle+abs (146,nt)':28s} {nt:>8} {1000*t64:>9.2f} {1000*t32:>9.2f} {t64/t32:>7.1f}x")
    print()
