import numba as nb


@nb.njit
def forward_eulr(f, x, u, dt, args):
    return x + dt * f(x, u, *args)
