import math

import matplotlib.pyplot as plt


def exact(x):
    return -2 + math.exp(-x) + math.exp(x) - 2.8 * x + 2.8 * x ** 2


def fill_A(a, b, n):
    res = []
    h = (b - a) / n
    for i in range(n + 1):
        if i == 0:
            res.append([1] + [0 for _ in range(n)])
        elif i == n:
            pres = [0 for _ in range(n + 1)]
            pres[i - 1] = -2
            pres[i] = 2 * h + 2 + h ** 2
            res.append(pres)
        else:
            pres = [0 for _ in range(n + 1)]
            pres[i - 1] = 1
            pres[i] = -2 - h ** 2
            pres[i + 1] = 1
            res.append(pres)
    return res


def fill_b(a, b, n):
    res = []
    h = (b - a) / n
    for i in range(n + 1):
        x = a + i * h
        if i == 0:
            res.append(0)
        elif i == n:
            ans = (2 * math.e + 0.8) * h * 2 - (7.6 + 2.8 * x * (1 - x)) * (h ** 2)
            res.append(ans)
        else:
            res.append((7.6 + 2.8 * x * (1 - x)) * (h ** 2))
    return res


def solve_matrix(a, b):
    n = len(a)
    x = [0 for _ in range(0, n)]

    v = [0 for _ in range(0, n)]
    u = [0 for _ in range(0, n)]

    v[0] = a[0][1] / (-a[0][0])
    u[0] = (- b[0]) / (-a[0][0])
    for i in range(1, n - 1):
        v[i] = a[i][i + 1] / (-a[i][i] - a[i][i - 1] * v[i - 1])
        u[i] = (a[i][i - 1] * u[i - 1] - b[i]) / (-a[i][i] - a[i][i - 1] * v[i - 1])
    v[n - 1] = 0
    u[n - 1] = (a[n - 1][n - 2] * u[n - 2] - b[n - 1]) / (-a[n - 1][n - 1] - a[n - 1][n - 2] * v[n - 2])

    x[n - 1] = u[n - 1]
    for i in range(n - 1, 0, -1):
        x[i - 1] = v[i - 1] * x[i] + u[i - 1]

    return x


def tdm(a, b, n):
    step = (b - a) / n
    x_list = [step * i for i in range(n + 1)]
    y_list = solve_matrix(fill_A(a, b, n), fill_b(a, b, n))
    return x_list, y_list


def exact_res(a, b):
    n = 10000
    exact_step = (b - a) / n
    exact_x = [exact_step * i for i in range(n)]
    exact_y = [exact(x) for x in exact_x]
    return exact_x, exact_y


def main():
    a, b = 0, 1
    n = 10

    plt.rcParams['figure.figsize'] = (20, 10)

    exact_x, exact_y = exact_res(a, b)
    tdm_x, tdm_y = tdm(a, b, n)

    plt.plot(exact_x, exact_y, label='Точное решение', color='red')
    plt.plot(tdm_x, tdm_y, label='Метод прогонки')
    plt.show()


if __name__ == '__main__':
    main()
