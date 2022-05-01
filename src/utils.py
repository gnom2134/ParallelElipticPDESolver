import numpy as np
from typing import Callable, NoReturn, Tuple
from datetime import datetime
from tqdm import tqdm


def fill_with_func(grid: np.array, x_0: float, y_0: float, h: float, func: Callable) -> NoReturn:
    x = x_0
    y = y_0
    for i in range(grid.shape[0]):
        x += h
        for j in range(grid.shape[1]):
            y += h
            grid[i, j] = func(x, y)
        y = y_0


def is_close(a: np.array, b: np.array, eps: float = 1e-6):
    return np.all(np.abs(a - b) < eps)


def calc_time(func: Callable, args: Tuple, its: int) -> Tuple[int, np.array]:
    dtime = []
    res = None
    for _ in tqdm(range(its)):
        st = datetime.now()
        res = func(*args)
        time_diff = datetime.now() - st
        dtime.append(time_diff.total_seconds() * 1e6 + time_diff.microseconds)
    return int(np.mean(dtime)), res
