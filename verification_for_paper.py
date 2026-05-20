from numba import njit
import numpy as np


@njit()
def build_grid_hash_from_sorted(cX, epsilon=1e-9):

    n = len(cX)

    x_min = cX[0]
    x_max = cX[-1]


    ideal_width = (x_max - x_min) / n if n > 1 else 2 * epsilon
    bucket_width = max(ideal_width, 2 * epsilon)


    min_bid = int(np.floor(x_min / bucket_width))
    max_bid = int(np.floor(x_max / bucket_width))
    num_buckets = max_bid - min_bid + 1


    ptr = np.zeros(num_buckets + 1, dtype=np.int64)

    current_bucket_idx = 0
    for i in range(n):
        bid = int(np.floor(cX[i] / bucket_width)) - min_bid
        while current_bucket_idx < bid:
            current_bucket_idx += 1
            ptr[current_bucket_idx] = i

    for i in range(current_bucket_idx + 1, num_buckets + 1):
        ptr[i] = n


    return ptr, min_bid, bucket_width

@njit()
def binary_search(a, b, cY, cX, epsilon=1e-9):

    n = len(cY)
    m = len(cX)

    if n == 0:
        return True
    if m == 0:
        return False

    prev_y=0

    for i in range(n):
        target = (cY[i] - b) / a

        idx = np.searchsorted(cX[prev_y:-1], target)
        idx=idx+prev_y

        is_match = False

        if idx < m:
            if abs(cX[idx] - target) <= epsilon:
                prev_y=idx
                is_match = True

        if not is_match and idx > 0:
            if abs(cX[idx - 1] - target) <= epsilon:
                prev_y=idx-1
                is_match = True

        if not is_match:
            return False

    return True

@njit()
def hash_search(a, b, cY, cX, ptr, min_bid, bucket_width, epsilon=1e-9):
    n_buckets = len(ptr) - 1

    for i in range(len(cY)):
        x = (cY[i] - b) / a
        found_match = False

        bid = int(np.floor(x / bucket_width)) - min_bid

        if 0 <= bid < n_buckets:
            start = ptr[bid]
            end = ptr[bid + 1]
            count = end - start

            if count == 1:
                if abs(x - cX[start]) <= epsilon:
                    found_match = True
            elif 1 < count <= 4:
                for j in range(start, end):
                    if abs(x - cX[j]) <= epsilon:
                        found_match = True
                        break
            elif count > 4:
                low = start
                high = end
                while low < high:
                    mid = (low + high) // 2
                    if cX[mid] < x:
                        low = mid + 1
                    else:
                        high = mid
                if low < end and (cX[low] - x) <= epsilon:
                    found_match = True
                elif low > start and (x - cX[low - 1]) <= epsilon:
                    found_match = True

        if found_match:
            continue

        offset = x % bucket_width
        if offset <= epsilon:
            nbid = bid - 1
            if 0 <= nbid < n_buckets:
                start_nb, end_nb = ptr[nbid], ptr[nbid + 1]
                if start_nb < end_nb:  # Non-empty
                    if (x - cX[end_nb - 1]) <= epsilon:
                        found_match = True

        if not found_match and offset >= (bucket_width - epsilon):
            nbid = bid + 1
            if 0 <= nbid < n_buckets:
                start_nb, end_nb = ptr[nbid], ptr[nbid + 1]
                if start_nb < end_nb:  # Non-empty
                    if (cX[start_nb] - x) <= epsilon:
                        found_match = True

        if not found_match:
            return False

    return True


@njit()
def verify(a,b,cX,cY,variant=0,hash_ptr=None, hash_min_bid=None, hash_bucket_width=None,epsilon=1e-9):
    ifcY=cY
    if a<0:
        ifcY=cY[::-1]
    if variant==0:
        return binary_search(a, b, ifcY, cX, epsilon)
    else:

        return hash_search(a, b, ifcY, cX, hash_ptr, hash_min_bid, hash_bucket_width, epsilon)
