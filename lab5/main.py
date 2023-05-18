from typing import Tuple, Callable

import numpy as np
from matplotlib import pyplot as plt

a, b = 0, 1
ranges = [10, 20, 30]
start_x = 0
start_y = 1


def f(x: float, y: float) -> float:
    return -2 * y + 2


def df(x: float, y: float) -> float:
    return 4 * y - 4


def d2f(x: float, y: float) -> float:
    return -8 * y + 8


def d3f(x: float, y: float) -> float:
    return 16 * y - 16


def res_y(x: float) -> float:
    return 1


def get_step(start: float, end: float, n: int):
    return (end - start) / n


def euler_with_recount(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x = start_x
    y = start_y
    step = get_step(a, b, n)
    res = []
    for i in range(0, n):
        next_x = x + step
        next_y = y + step / 2 * (f(x, y) + f(next_x, y + step * f(x, y)))
        res.append((next_x, next_y))
        x = next_x
        y = next_y
    return np.array(list(map(lambda li: li[0], res))), np.array(list(map(lambda li: li[1], res)))


def implicit_euler(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x = start_x
    y = start_y
    step = get_step(a, b, n)
    res = []
    for i in range(0, n):
        next_x = x + step
        next_y = (y + 2 * step) / (1 + 2 * step)
        res.append((next_x, next_y))
        x = next_x
        y = next_y
    return np.array(list(map(lambda li: li[0], res))), np.array(list(map(lambda li: li[1], res)))


def taylor(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x = start_x
    y = start_y
    step = get_step(a, b, n)
    res = []
    for i in range(0, n):
        next_x = x + step
        next_y = y + step * f(x, y) + (step ** 2) * df(x, y) / 2 + (step ** 3) * d2f(x, y) / 6 + step ** 4 * d3f(x, y) / 24
        res.append((next_x, next_y))
        x = next_x
        y = next_y
    return np.array(list(map(lambda li: li[0], res))), np.array(list(map(lambda li: li[1], res)))


def format_list(name: str, values: np.ndarray) -> str:
    val_str = ', '.join(map(str, values))
    return f'{name}=[{val_str}]'


def test_method(name: str, func: Callable[[int], Tuple[np.ndarray, np.ndarray]]) -> None:
    print(name, ':')
    for n in ranges:
        print(f'{n=}')
        x_list, y_list = func(n)
        print(format_list('x_i', x_list))
        print(format_list('y_i', y_list))
        print('-' * 20)


def draw_result(name: str, func: Callable[[int], Tuple[np.ndarray, np.ndarray]], n: int) -> None:
    x_list, y_list = func(n)
    plt.rcParams['figure.figsize'] = (20, 10)
    exact_y = [res_y(x) for x in x_list]
    plt.plot(x_list, exact_y, label='Точное решение', color='black')
    plt.plot(x_list, y_list, label=name)
    plt.show()


def main():
    test_method('Метод Эйлера с пересчетом', euler_with_recount)
    test_method('Неявный метод Эйлера', implicit_euler)
    test_method('Метод Тейлора 4го порядка', taylor)
    # Можно нарисовать график для определенного метода с заданным n
    # draw_result('Метод Эйлера с пересчетом', taylor, 30)


if __name__ == '__main__':
    main()
