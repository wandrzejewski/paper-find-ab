import time
from contextlib import contextmanager
import numpy as np

from algs_for_paper import old_solution, new_solution, range_chooser, range_chooser_no_amin_amax
from verification_for_paper import build_grid_hash_from_sorted


@contextmanager
def timer(label):
    print(f"\n[{label}] Starting time measurement")
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"[{label}] Elapsed: {end - start:.6f}s {(end - start)*1000000000:.6f}ns")



def generate_no_solution(N,low, high,M,a,b):
    cX = np.random.uniform(low, high, size=N)
    cY = a*np.random.uniform(low, high, size=M)+b
    return cX,cY

def generate_with_solution(N,low, high,M,a,b):
    cX = np.random.uniform(low, high, size=N)
    indices=np.random.choice(np.arange(0,N), size=M, replace=False);
    cY = a*cX[indices]+b
    return cX, cY



if __name__ == "__main__":
    #warmup might be needed so jit compiles all procedures

    with timer("Example 1 OLD algorithm, no solution"):
        cX,cY=generate_no_solution(20000,-1,1,5000,10,15)
        cX=np.sort(cX)
        cY=np.sort(cY)
        a,b,_,_ = old_solution(cX,cY,0,[0],0,0)
        print("a,b=",a,",",b)
    with timer("Example 2 OLD algorithm, with solution"):
        cX,cY=generate_with_solution(20000,-1,1,5000,10,15)
        cX=np.sort(cX)
        cY=np.sort(cY)
        a,b,_,_ = old_solution(cX,cY,0,[0],0,0)
        print("a,b=", a, ",", b)

    with timer("Example 3 NEW algorithm, no amin, amax, no solution"):
        cX,cY=generate_no_solution(20000,-1,1,5000,10,15)
        cX=np.sort(cX)
        cY=np.sort(cY)
        hash_ptr,hash_min_bid,hash_bucket_width=build_grid_hash_from_sorted(cX)
        a,b,_,_ = new_solution(cX,cY,0,0,range_chooser_no_amin_amax,1,hash_ptr,hash_min_bid,hash_bucket_width)
        print("a,b=",a,",",b)

    with timer("Example 3 NEW algorithm, no amin, amax, with solution"):
        cX,cY=generate_with_solution(20000,-1,1,5000,10,15)
        cX = np.sort(cX)
        cY = np.sort(cY)
        hash_ptr, hash_min_bid, hash_bucket_width = build_grid_hash_from_sorted(cX)
        a, b, _, _ = new_solution(cX, cY, 0, 0, range_chooser_no_amin_amax, 1, hash_ptr, hash_min_bid, hash_bucket_width)
        print("a,b=", a, ",", b)

    with timer("Example 5 NEW algorithm, with amin, amax, no solution"):
        cX, cY = generate_no_solution(20000, -1, 1, 5000, 10, 15)
        cX = np.sort(cX)
        cY = np.sort(cY)
        hash_ptr, hash_min_bid, hash_bucket_width = build_grid_hash_from_sorted(cX)
        a, b, _, _ = new_solution(cX, cY, 5, 15, range_chooser, 1, hash_ptr, hash_min_bid,
                                  hash_bucket_width)
        print("a,b=", a, ",", b)

    with timer("Example 6 NEW algorithm, with amin, amax, with solution"):
        cX, cY = generate_with_solution(20000, -1, 1, 5000, 10, 15)
        cX = np.sort(cX)
        cY = np.sort(cY)
        hash_ptr, hash_min_bid, hash_bucket_width = build_grid_hash_from_sorted(cX)
        a, b, _, _ = new_solution(cX, cY, 5, 15, range_chooser, 1, hash_ptr, hash_min_bid,
                                  hash_bucket_width)
        print("a,b=", a, ",", b)