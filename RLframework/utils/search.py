from bisect import bisect_right
import numpy as np


def find_le(a, x):
    'Find rightmost value less than x'
    i = bisect_right(a, x)
    if i:
        return i-1
    raise ValueError


def find_le_serialized(a, x):
    _a = list(np.linspace(0, 1.1, len(a)))
    i = bisect_right(_a, x)
    if i:
        return a[i-1]


def _serialize(x):
    return list(np.linspace(min(x), max(x), len(x)))


if __name__ == "__main__":
    print(find_le([1, 2, 3, 5, 6, 10, 15, 30], 30))
    print(find_le_serialized([1, 2, 3, 5, 6, 10, 15, 30], 0.99))
