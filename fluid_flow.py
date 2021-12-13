import numpy as np
from numba import jit
from parameters import *
from numpy import pi, sin
from Funcs import df1_2, df1_2
from poisson_solver import SOR_solver

w = 2 / (1 + sin(pi/N)) # optimal SOR parameter in the symmatrical case NxN

@jit(nopython=True)
def set_boundary(u,v):

    #left wall (slipping)
    u[0,:] = 0
    v[0,:] = v[1,:]
    #u[M-1,:] = 0
    #v[M-1,:] = v[M-2,:]

    #gas inlets (upper and lower left boundaries)
    u[:Nc_lw, 0] = 0 # flow/2
    u[:Nc_lw,-1] = 0 #flow/2
    v[:Ns_c, 0]  =  Uslot
    v[:Ns_c,-1]  = -Uslot
    v[Ns_c:Nc_lw, 0] =  Ucoflow
    v[Ns_c:Nc_lw,-1] = -Ucoflow

    # upper and lower right non-slipping walls:
    u[Nc_lw:, 0] = 0 # u[Nc_lw:, 1]
    v[Nc_lw:, 0] = 0
    u[Nc_lw:,-1] = 0 # u[Nc_lw:,-2]
    v[Nc_lw:,-1] = 0

    # right gaz outlet (forced steady-state --> Neumann BC)
    u[M-1,:] = u[M-2,:]
    v[M-1,:] = v[M-2,:]

    return u,v


@jit(nopython=True)
def compute_P(u, v, dx, dt, rho, Pprev=None):
    # no need to compute dvdx or dudy
    dudx = df1_2(u, dx, axis=0)
    dvdy = df1_2(v, dx, axis=1)
    b = dx**2 * rho / dt * (dudx + dvdy)

    return SOR_solver(b, Pprev=Pprev, w=w, maxit=10000)

@jit(nopython=True)
def advance_fluid_flow(Nt, u, v, f, dt):
    """
    f: Time integration function, e.g. advance_adv_diff_RK3
    """
    P = np.zeros_like(u)

    for n in range(Nt):

        u = f(u, dt, u, v, dx, dy, nu)
        v = f(v, dt, u, v, dx, dy, nu)
        u,v = set_boundary(u,v)

        # NOTE: using Pprev here increases Poisson solver convergence speed a lot!
        P = compute_P(u, v, dx, dt, rho, Pprev=P)

        # third step (P)
        dPdx = df1_2(P, dx, axis=0)
        dPdy = df1_2(P, dx, axis=1)

        u = u - dt / rho * dPdx
        v = v - dt / rho * dPdy

        # apply BCs one more at the end?
        #u,v = set_boundary(u,v)
    return u, v
