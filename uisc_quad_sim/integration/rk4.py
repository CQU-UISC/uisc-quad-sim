import numba as nb


@nb.njit
def rk4(f, x, u, dt, args):
    k1 = f(x, u, *args)
    k2 = f(x + 0.5 * dt * k1, u, *args)
    k3 = f(x + 0.5 * dt * k2, u, *args)
    k4 = f(x + dt * k3, u, *args)
    final = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return final
