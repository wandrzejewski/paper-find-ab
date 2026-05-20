from numba import njit
import numpy as np

from verification_for_paper import verify


@njit(inline='always')
def compute_ab(x1,y1,x2,y2):
    a=(y1-y2)/(x1-x2)
    b=y1-a*x1
    return a,b


@njit()
def estimate_ab_stat(cX,cY):
    ex=np.mean(cX)
    ey=np.mean(cY)
    sx=np.std(cX)
    sy=np.std(cY)

    a=sy/sx
    b=ey-a*ex

    return a,b


@njit()
def find_closest_index(sorted_array, target):
    idx = np.searchsorted(sorted_array, target)

    if idx == 0:
        return 0
    if idx == len(sorted_array):
        return len(sorted_array) - 1

    left_val = sorted_array[idx - 1]
    right_val = sorted_array[idx]

    if abs(target - left_val) <= abs(target - right_val):
        return idx - 1
    else:
        return idx


@njit()
def old_solution(cX, cY, variant=0, hash_ptr=None, hash_min_bid=None, hash_bucket_width=None):
    j1=0
    j2=len(cY)-1
    y1=cY[j1]
    y2=cY[j2]

    start=0
    end=len(cX)-1

    while start<=end:
        iters=[start,end]
        if start == end:
            iters.pop()
        for i1 in iters:
            x1 = cX[i1]

            for i2 in range(len(cX)):

                if i1!=i2:
                    x2 = cX[i2]
                    a, b = compute_ab(x1, y1, x2, y2)
                    if abs(a)<200e6:
                        vr=verify(a, b, cX, cY, variant, hash_ptr, hash_min_bid, hash_bucket_width)
                        if vr:
                            return a, b, i1, i2

        start+=1
        end-=1

    return None,None,None,None




@njit()
def range_chooser_no_amin_amax(cX, len_cY, y1, y2, i1, amin, amax,slope_sign):
    N = len(cX)
    idx_start_wide=0
    idx_end_wide=N

    if slope_sign > 0:
        idx_start_wide = min(i1 + len_cY - 1, N)
    else:
        idx_end_wide = max(0, i1 - len_cY + 2)

    min_x2 = cX[0]
    max_x2 = cX[-1]

    return range(max(0, idx_start_wide), min(N, idx_end_wide)),min_x2,max_x2

@njit()
def range_chooser(cX, len_cY, y1, y2, i1, amin, amax, slope_sign):
    N = len(cX)
    idx_start_wide=0
    idx_end_wide=N

    if slope_sign > 0:
        idx_start_wide = min(i1 + len_cY - 1, N)
    else:
        idx_end_wide = max(0, i1 - len_cY + 2)

    dy = y2 - y1
    x1 = cX[i1]


    min_x2 = cX[0]
    max_x2 = cX[-1]
    if amax is not None and amin is not None:
        min_dx = dy / amax
        max_dx = dy / amin if amin > 0 else float('inf')

        if slope_sign > 0:
            min_x2 = x1 + min_dx
            max_x2 = x1 + max_dx
        else:
            min_x2 = x1 - max_dx
            max_x2 = x1 - min_dx

    return range(max(0, idx_start_wide), min(N, idx_end_wide)),min_x2,max_x2



@njit()
def new_solution(cX, cY, amin, amax, range_chooser, variant=0, hash_ptr=None, hash_min_bid=None, hash_bucket_width=None):

    a_est, b_est = estimate_ab_stat(cX, cY)

    y1 = cY[0]
    y2 = cY[-1]
    len_cY = len(cY)
    N = len(cX)

    x1p=(y1-b_est)/a_est
    x2p=(y2-b_est)/a_est

    i1p = find_closest_index(cX,x1p)
    i2p = find_closest_index(cX,x2p)
    i1p = int(i1p)
    i2p = int(i2p)

    max_r = max(i1p, N - 1 - i1p, i2p, N - 1 - i2p)
    min_r=max(0, (len(cY) - abs(i2p - i1p)) // 2)
    for r in range(min_r,max_r + 1):
        for slope_sign, c1, c2 in [(1, i1p, i2p), (-1, i2p, i1p)]:
            for step, i1 in enumerate((c1 - r, c1 + r)):
                if r == 0 and step == 1:
                    continue
                if 0 <= i1 < N:
                    valid_range,min_x2,max_x2=range_chooser(cX, len_cY, y1, y2, i1, amin, amax, slope_sign)
                    start = max(valid_range.start, c2 - r, 0)
                    end = min(valid_range.stop, c2 + r + 1, N)

                    x1 = cX[i1]

                    for i2 in range(start, end):
                        x2=cX[i2]
                        if min_x2<=x2 <= max_x2:
                            a, b = compute_ab(x1, y1, x2, y2)
                            vr = verify(a, b, cX, cY, variant, hash_ptr, hash_min_bid, hash_bucket_width)
                            if vr:
                                return a, b, i1, i2
                        if x2 > max_x2:
                            break

            if r > 0:
                i1_start_idx = max(0, c1 - r + 1)
                i1_end_idx = min(N, c1 + r)


                if slope_sign > 0:
                    i1_end_idx = min(i1_end_idx, N - len(cY) + 1)
                elif slope_sign < 0:
                    i1_start_idx = max(i1_start_idx, len(cY) - 1)

                for i1 in range(i1_start_idx, i1_end_idx):
                    valid_range,min_x2,max_x2=range_chooser(cX, len_cY, y1, y2, i1, amin, amax, slope_sign)
                    x1=cX[i1]

                    for i2 in (c2 - r, c2 + r):
                        if 0 <= i2 < N and i2 in valid_range:
                            x2=cX[i2]

                            if min_x2 <= x2 <= max_x2:
                                a,b=compute_ab(x1,y1,cX[i2],y2)
                                vr = verify(a, b, cX, cY, variant, hash_ptr, hash_min_bid, hash_bucket_width)
                                if vr:
                                    return a, b, i1, i2

                            if x2 > max_x2:
                                break

    return None, None, None, None


