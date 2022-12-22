from typing import List, Tuple, Union

import numpy as np

MATRIX = np.array([[-0.12, 1.58, 1.13, 4.73], [0.92, -0.80, 0.11, 1.37], [-0.28, -0.22, -0.60, -2.58]])

A = np.array([[0.92, -0.80, 0.11], [-0.12, 1.58, 1.13], [-0.28, -0.22, -0.60]])

B = np.array([1.37, 4.73, -2.58]).T


def gauss(matrix: np.ndarray) -> List[float]:
    n = matrix.shape[0]
    s = np.copy(matrix)

    for k in np.arange(n):
        s[k, :] /= matrix[k, k]
        for i in np.arange(k + 1, n):
            K = s[i, k] / s[k, k]
            s[i] -= s[k] * K
        matrix = np.copy(s)

    for k in np.arange(n - 1, -1, -1):
        s[k, :] /= matrix[k, k]
        for i in np.arange(k - 1, -1, -1):
            K = s[i, k] / s[k, k]
            s[i] -= s[k] * K

    return [s[i, n] for i in range(n)]


def change_order_of_lines(matrix: np.ndarray, col: int) -> None:
    ind = col + np.argmax(np.abs(matrix[col:, col]))
    if ind != col:
        matrix[col, :], matrix[ind, :] = np.copy(matrix[ind, :]), np.copy(matrix[col, :])


def gauss_with_choose(matrix: np.ndarray) -> List[float]:
    n = matrix.shape[0]
    for k in np.arange(n - 1):
        change_order_of_lines(matrix, k)
        for i in np.arange(k + 1, n):
            K = matrix[i, k] / matrix[k, k]
            matrix[i, :] -= matrix[k, :] * K

    if np.any(np.diag(matrix) == 0):
        return []

    answer = np.zeros(n)
    for i in range(n - 1, -1, -1):
        answer[i] = (matrix[i][-1] - sum([matrix[i][j] * answer[j] for j in range(i + 1, n)])) / matrix[i][i]
    return list(answer)


def jacobi_step(revD: np.ndarray, R: np.ndarray, x_current: np.ndarray, b: np.ndarray) -> np.ndarray:
    return -revD @ R @ x_current + revD @ b


def jacobi(A: np.ndarray, b: np.ndarray, eps: float = 5e-5) -> Tuple[np.ndarray, int]:
    D = np.diag(A)
    R = A - np.diagflat(D)
    revD = np.linalg.inv(np.diagflat(D))

    x_current = np.array([a / A[i][i] for i, a in enumerate(b)])
    x_next = jacobi_step(revD, R, x_current, b)
    count = 0
    while np.linalg.norm(x_next - x_current) > eps:
        x_current = x_next
        x_next = jacobi_step(revD, R, x_current, b)
        count += 1
    return x_next, count


def seidel_step(DL: np.ndarray, R: np.ndarray, x_current: np.ndarray, b: np.ndarray) -> np.ndarray:
    return -DL @ R @ x_current + DL @ b


def seidel(A: np.ndarray, b: np.ndarray, eps: float = 5e-5) -> Tuple[List[float], int]:
    D = np.diagflat(np.diag(A))
    L = np.tril(A, k=-1)
    R = np.triu(A, k=1)
    DL = np.linalg.inv(D + L)

    x_current = np.zeros(A.shape[0])
    x_next = -DL @ R @ x_current + DL @ b
    count = 0
    while np.linalg.norm(x_next - x_current) > eps:
        x_current = x_next
        x_next = -DL @ R @ x_current + DL @ b
        count += 1
    return x_next, count


def print_answer(ans: Union[List[float], np.ndarray]) -> None:
    for i, res in enumerate(ans):
        print(f'\tx{i + 1}={res}')


def main() -> None:
    print('Метод Гаусса:')
    print_answer(gauss(MATRIX))
    print('Метод Гаусса с выбором главного элемента по всей матрице:')
    print_answer(gauss_with_choose(MATRIX))
    print('Метод Якоби:')
    ans, iterations = jacobi(A, B)
    print(f'    Количество итераций: {iterations}')
    print_answer(ans)
    print('Метод Гаусса-Зейделя:')
    ans, iterations = seidel(A, B)
    print(f'    Количество итераций: {iterations}')
    print_answer(ans)


if __name__ == '__main__':
    main()
