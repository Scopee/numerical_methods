from math import exp, log
from typing import Tuple, Callable, Any

eps = 0.5e-5
default_interval = (2, 2.5)


def get_start_x(f: Callable[[float], float], d2f: Callable[[float], float], interval: Tuple[float, float]) -> float:
    a, b = interval
    if f(b) * d2f(b) > 0:
        return b
    elif f(a) * d2f(a) > 0:
        return a


def func(x: float) -> float:
    return exp(x) - 4.6 * x


def func_der(x: float) -> float:
    return exp(x) - 4.6


def func_second_der(x: float) -> float:
    return exp(x)


def phi(x: float) -> float:
    return log(4.6 * x)


def half_div(f: Callable[[float], float], interval: Tuple[float, float]) -> Tuple[float, int]:
    n = 0
    a, b = interval
    x = None
    while abs(b - a) >= eps:
        x = abs(a + b) / 2
        if f(x) * f(a) < 0:
            b = x
        elif f(x) * f(b) < 0:
            a = x
        n += 1
    return x, n


def newton(f: Callable[[float], float], df: Callable[[float], float],
           d2f: Callable[[float], float], interval: Tuple[float, float]) -> Tuple[float, int]:
    n = 0
    start_x = get_start_x(f, d2f, interval)
    x = start_x - f(start_x) / df(start_x)
    while abs(x - start_x) >= eps:
        start_x = x
        x = x - f(x) / df(x)
        n += 1
    return x, n


def newton_modified(f: Callable[[float], float], df: Callable[[float], float],
                    d2f: Callable[[float], float], interval: Tuple[float, float]) -> Tuple[float, int]:
    n = 0
    start_x = get_start_x(f, d2f, interval)
    x_1 = start_x
    x = start_x - f(start_x) / df(start_x)
    while abs(x - x_1) >= eps:
        x_1 = x
        x = x - f(x) / df(start_x)
        n += 1
    return x, n


def chord(f: Callable[[float], float], interval: Tuple[float, float]) -> Tuple[float, int]:
    n = 0
    a, x_p = interval
    x_c = a - (f(a) * (x_p - a)) / (f(x_p) - f(a))
    while abs(x_c - x_p) >= eps:
        x_p = x_c
        x_c = a - (f(a) * (x_p - a)) / (f(x_p) - f(a))
        n += 1
    return x_c, n


def mov_chord(f: Callable[[float], float], interval: Tuple[float, float]) -> Tuple[float, int]:
    n = 0
    a, b = interval
    x_c = b - (f(b) * (b - a)) / (f(b) - f(a))
    while abs(b - a) >= eps:
        a = b
        b = x_c
        x_c = b - (f(b) * (b - a)) / (f(b) - f(a))
        n += 1
    return x_c, n


def simple_iter(f: Callable[[float], float], interval: Tuple[float, float]) -> Tuple[float, int]:
    n = 0
    a, b = interval
    x_n = a
    x_c = f(x_n)
    while abs(x_c - x_n) >= eps:
        x_n = x_c
        x_c = f(x_n)
        n += 1
    return x_c, n


def print_results(name: str, f: Callable[[Any], Tuple[float, int]], *args):
    print(f'{name}:')
    x, n = f(*args)
    print(f'{x=}, {n=}')


def main() -> None:
    print_results("Метод половинного деления", half_div, func, default_interval)
    print_results("Метод Ньютона", newton, func, func_der, func_second_der, default_interval)
    print_results("Метод Ньютона модифицированный", newton_modified, func, func_der, func_second_der, default_interval)
    print_results("Метод хорд", chord, func, default_interval)
    print_results("Метод подвижных хорд", mov_chord, func, default_interval)
    print_results("Метод простой итерации", simple_iter, phi, default_interval)


if __name__ == '__main__':
    main()

