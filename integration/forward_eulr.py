def forward_eulr(f,x,u,dt):
    return x + dt*f(x,u)