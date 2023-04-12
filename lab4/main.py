import math
from typing import Callable, Tuple, List, Optional, Any


def func(x: float) -> float:
    return math.sin(math.e ** (x / 3) + x)


def func_der(x: float) -> float:
    return 1 / 3 * (math.e ** (x / 3) + 3) * math.cos(x + math.e ** (x / 3))


def simpson(f: Callable[[float], float], step: float, interval: Tuple[float, float], prev_res: Optional[float],
            f_derr: Callable[[float], float] = None) -> Tuple[float, float]:
    a, b = interval

    n = int((b - a) / step)
    res = 0
    for i in range(n):
        x_i = a + i * step
        x_i12 = half_step(x_i, step)
        x_i1 = x_i + step
        res += f(x_i) + 4 * f(x_i12) + f(x_i1)
    res = (step / 6) * res
    R = None
    if prev_res:
        R = abs(res - prev_res) / 15
    return res, R


def euler(f: Callable[[float], float], step: float, interval: Tuple[float, float], prev_res: Optional[float], f_derr: Callable[[float], float]) -> \
        Tuple[float, float]:
    a, b = interval
    res = step / 2 * (f(a) + f(b)) + (step ** 2) / 12 * (f_derr(a) - f_derr(b))

    n = int((b - a) / step)
    s = 0
    for i in range(1, n):
        x_i = a + i * step
        s += f(x_i)
    res += step * s
    R = None
    if prev_res:
        R = abs(res - prev_res) / 15
    return res, R


def half_step(x: float, step: float) -> float:
    return x + step / 2


def test_method(
        test: Callable[[Callable[[float], float], float, Tuple[float, float], Optional[float], Callable[[float], float]], Tuple[float, float]],
        steps: List[float],
        method_name: str, *args: Any):
    i = (0, 1)
    prev_res = None
    for step in steps:
        print(f'{method_name} с шагом {step}:')
        res, R = test(func, step, i, prev_res, *args)
        if not R:
            prev_res = res
        print(f'I = {res}')
        if R:
            print(f'Погрешность по Рунге:')
            print(f'R = {R}')
        print('-' * 20)
    print('*' * 20)


def gauss(f: Callable[[float], float]):
    A = 5 / 18
    B = 4 / 9
    C = 5 / 18
    x0 = (5 - math.sqrt(15)) / 10
    x1 = 1 / 2
    x2 = (5 + math.sqrt(15)) / 10
    return A * f(x0) + B * f(x1) + C * f(x2)


if __name__ == '__main__':
    test_method(simpson, [0.1, 0.05, 0.025], 'Метод Симпсона')
    test_method(euler, [0.1, 0.05, 0.025], 'Метод Эйлера', func_der)

    print('Метод Гаусса по 3 узлам')
    print(gauss(func))
