import numpy as np
from matplotlib import pyplot as plot
import scipy.integrate as integrate


def integral(f, a, b):
    return integrate.trapezoid([f(x) for x in np.linspace(a, b, 10000)], dx=(b - a) / 10000)


def x_i(i, n):
    return (2 / n) * i


# galerkin method

def e(i, x, n):
    range_length = 2 / n
    if x_i(i - 1, n) <= x <= x_i(i, n):
        return (x - x_i(i - 1, n)) / range_length

    elif x_i(i, n) < x <= x_i(i + 1, n):
        return (x_i(i + 1, n) - x) / range_length
    return 0


def e_prim(i, x, n):
    range_length = 2 / n
    if x_i(i - 1, n) <= x <= x_i(i, n):
        return 1 / range_length
    elif x_i(i, n) < x <= x_i(i + 1, n):
        return -1 / range_length
    return 0


def L(v, n):
    f1 = lambda x: np.sin(x) * e(v, x, n)
    return integral(f1, 0, 2)


def B(u, v, n):
    f1 = lambda x: e_prim(u, x, n) * e_prim(v, x, n)
    f2 = lambda x: e(u, x, n) * e(v, x, n)
    return integral(f1, 0, 2) - integral(f2, 0, 2) - (e(u, 2, n) * e(v, 2, n))


def u(u_i, x, n):
    result = 0
    for i in range(1, len(u_i) + 1):
        result += u_i[i - 1] * e(i, x, n)
    return result


def main():
    n = int(input("n: "))
    B_matrix = np.zeros((n, n))
    L_matrix = np.zeros(n)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            B_matrix[i - 1, j - 1] = B(i, j, n)

    for i in range(1, n + 1):
        L_matrix[i - 1] = L(i, n)

    u_i = np.linalg.solve(B_matrix, L_matrix)
    x = np.linspace(0, 2, 100)
    values = [u(u_i, x_i, n) for x_i in x]

    plot.plot(x, values)
    plot.title("Wykres u(x)")
    plot.xlabel("x")
    plot.ylabel("u(x)")
    plot.grid(True)
    plot.show()


main()
