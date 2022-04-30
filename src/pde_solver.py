import numpy as np
from typing import Tuple, List
from utils import fill_with_func, is_close, calc_time
from multiprocessing import Pool
from scipy.sparse import lil_matrix, linalg


def example_f(x: float, y: float) -> float:
    return (-(x**2) - y**2) * np.exp(x * y)


def build_K(
    C_number: int, I_number: int, number_of_parts: int
) -> Tuple[np.array, List[np.array], List[np.array], List[np.array]]:
    interface_points_num = C_number // (number_of_parts - 1)

    K_C = lil_matrix((C_number, C_number), dtype=np.float64)
    K_C.setdiag([4 for _ in range(C_number)])
    rng = np.arange(C_number - 1)
    K_C[rng, rng + 1] = -1
    K_C[
        rng[interface_points_num - 1 :: interface_points_num],
        rng[interface_points_num - 1 :: interface_points_num] + 1,
    ] = 0
    K_C[rng + 1, rng] = -1
    K_C[
        rng[interface_points_num - 1 :: interface_points_num] + 1,
        rng[interface_points_num - 1 :: interface_points_num],
    ] = 0

    K_IC = []
    K_CI = []
    K_I = []
    for i in range(number_of_parts):
        K_I_i = lil_matrix((I_number, I_number), dtype=np.float64)
        K_I_i.setdiag([4 for _ in range(I_number)])
        rng = np.arange(I_number - 1)
        K_I_i[rng, rng + 1] = -1
        K_I_i[
            rng[interface_points_num - 1 :: interface_points_num],
            rng[interface_points_num - 1 :: interface_points_num] + 1,
        ] = 0
        K_I_i[rng + 1, rng] = -1
        K_I_i[
            rng[interface_points_num - 1 :: interface_points_num] + 1,
            rng[interface_points_num - 1 :: interface_points_num],
        ] = 0
        rng = np.arange(I_number - interface_points_num)
        K_I_i[rng, rng + interface_points_num] = -1
        K_I_i[rng + interface_points_num, rng] = -1

        K_CI_i = lil_matrix((C_number, I_number), dtype=np.float64)
        rng = np.arange(interface_points_num)
        if i > 0:
            K_CI_i[rng + (i - 1) * interface_points_num, rng] = -1
        if i < number_of_parts - 1:
            K_CI_i[rng + i * interface_points_num, -rng[::-1] - 1] = -1

        K_CI.append(K_CI_i)
        K_IC.append(K_CI_i.T)
        K_I.append(K_I_i)

    return K_C, K_I, K_IC, K_CI


def pde_solve_brute(f_value: np.array) -> np.array:
    h, w = f_value.shape
    f_value = f_value.T.reshape((-1,))

    # building K
    K = lil_matrix((f_value.shape[0], f_value.shape[0]), dtype=np.float64)
    K.setdiag([4 for _ in range(f_value.shape[0])])
    rng = np.arange(f_value.shape[0] - 1)
    K[rng, rng + 1] = -1
    K[
        rng[h - 1 :: h],
        rng[h - 1 :: h] + 1,
    ] = 0
    K[rng + 1, rng] = -1
    K[
        rng[h - 1 :: h] + 1,
        rng[h - 1 :: h],
    ] = 0
    rng = np.arange(f_value.shape[0] - h)
    K[rng, rng + h] = -1
    K[rng + h, rng] = -1

    return linalg.inv(K) @ f_value


def step1(args):
    s, K_CI, K_I_inv, f_value, C_size, I_size = args
    return K_CI[s] @ K_I_inv @ f_value[C_size + s * I_size : C_size + (s + 1) * I_size]


def step2(args):
    s, K_CI, K_I_inv = args
    return K_CI[s] @ K_I_inv @ K_CI[s].T


def step3(args):
    s, K_I_inv, f_value, K_IC, u_C, C_size, I_size = args
    return K_I_inv @ (f_value[C_size + s * I_size : C_size + (s + 1) * I_size] - K_IC[s] @ u_C)


def pde_solve_parallel(f_value: np.array, divide_on: int = 2, processes: int = 8):
    h, w = f_value.shape
    if (w + 1) % divide_on != 0:
        raise AttributeError(f"Can not divide in {divide_on} equal domains")
    I_size = (h * w - (divide_on - 1) * h) // divide_on
    C_size = (divide_on - 1) * h
    K_C, K_I, K_IC, K_CI = build_K(C_size, I_size, divide_on)

    # reshape f
    indices = []
    step = (w - divide_on + 1) // divide_on
    for i in range(step, w, step + 1):
        indices.append(i)
    for i in range(0, w, 1):
        if (i - step) % (step + 1) == 0:
            continue
        indices.append(i)
    back_indices = [0 for x in range(len(indices))]
    for i in range(len(indices)):
        back_indices[indices[i]] = i
    f_value = f_value[:, indices].T.reshape((-1,))

    K_I_inv = linalg.inv(K_I[0])

    executors = None
    if processes > 1:
        executors = Pool(processes)

    g = f_value[:C_size].copy()
    if processes > 1:
        res = executors.map(step1, [(x, K_CI, K_I_inv, f_value, C_size, I_size) for x in range(divide_on)])
        for i in res:
            g -= i
    else:
        for s in range(divide_on):
            g -= K_CI[s] @ K_I_inv @ f_value[C_size + s * I_size : C_size + (s + 1) * I_size]

    S_C = K_C.copy().astype(np.float64)
    if processes > 1:
        res = executors.map(step2, [(x, K_CI, K_I_inv) for x in range(divide_on)])
        for i in res:
            S_C -= i
    else:
        for s in range(divide_on):
            S_C -= K_CI[s] @ K_I_inv @ K_CI[s].T

    u_C = linalg.inv(S_C) @ g

    if processes > 1:
        u_I = executors.map(step3, [(x, K_I_inv, f_value, K_IC, u_C, C_size, I_size) for x in range(divide_on)])
        u_I = np.array(list(u_I)).reshape((-1,))
    else:
        u_I = []
        for s in range(divide_on):
            u_I.append(K_I_inv @ (f_value[C_size + s * I_size : C_size + (s + 1) * I_size] - K_IC[s] @ u_C))
        u_I = np.array(u_I).reshape((-1,))

    return np.concatenate((u_C, u_I)).reshape((w, h)).T[:, back_indices].T.reshape((-1,))


if __name__ == "__main__":
    h = 1e-2
    height = 30
    width = 255
    x_0 = 0
    y_0 = 0
    split_into = 64
    average_its = 5

    grid = np.zeros((height, width))
    fill_with_func(grid, x_0, y_0, h, example_f)

    dtime1, res1 = calc_time(pde_solve_brute, (grid,), 1)

    dtime2, res2 = calc_time(pde_solve_parallel, (grid, split_into, 2), average_its)

    dtime3, res3 = calc_time(pde_solve_parallel, (grid, split_into, 4), average_its)

    dtime4, res4 = calc_time(pde_solve_parallel, (grid, split_into, 8), average_its)

    dtime5, res5 = calc_time(pde_solve_parallel, (grid, split_into, 1), average_its)

    assert is_close(res1, res2) and is_close(res1, res3) and is_close(res1, res4) and is_close(res1, res5)
    print(f"Grid size {height}x{width}, split into {split_into} chunks")
    print(f"Brute algorithm: {dtime1}")
    print(f"Parallel algorithm (2 ps): {dtime2}")
    print(f"Parallel algorithm (4 ps): {dtime3}")
    print(f"Parallel algorithm (8 ps): {dtime4}")
    print(f"Split algorithm (1 ps): {dtime5}")

    print(
        f"| {height}x{width} | {split_into} | {dtime1} | {dtime2} | {dtime3} | {dtime4} | {dtime5} |"
    )
