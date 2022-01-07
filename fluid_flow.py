import numpy as np
from numba import jit
from parameters import nu, Uslot, Ucoflow, rho, set_resolution
from math import pi, sin
from Funcs import df1_2, df1_2
from poisson_solver import SOR_solver

@jit(nopython=True)
def set_boundary(u,v, Ns_c, Nc_lw):

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
    u[-1,:] = u[-2,:]
    v[-1,:] = v[-2,:]

    return u,v


@jit(nopython=True)
def compute_P(u, v, dx, dt, rho, Pprev=None, w=1.5, atol=1e-4):
    # no need to compute dvdx or dudy
    dudx = df1_2(u, dx, axis=0)
    dvdy = df1_2(v, dx, axis=1)
    b = dx**2 * rho / dt * (dudx + dvdy)

    return SOR_solver(b, Pprev=Pprev, w=w, atol=atol, maxit=10000)

@jit(nopython=True)
def advance_fluid_flow(Nt, u, v, f, dt, w=None, atol=1e-4, P=None):
    """
    Evolve the fluid velocities u,v, according the incompressible N.-V.-Eqs
    over Nt time steps of duration dt.
    Note: the first time this function is executed takes long since numba needs to compile a lot).
    Args:
    - f: Time integration function, e.g. advance_adv_diff_RK3
    - P: Initial pressure field. Defaults to zeros if not provided.
    """
    if P is None:
        P = np.zeros_like(u)

    N, M = u.shape

    dx, dy, Ns_c, Nc_lw = set_resolution(N,M)

    if w is None:
        w = 2 / (1 + sin(pi/N)) # optimal SOR parameter in the symmatrical case NxN

    for n in range(Nt):

        u = f(u, dt, u, v, dx, dy, nu)
        v = f(v, dt, u, v, dx, dy, nu)
        u,v = set_boundary(u,v, Ns_c, Nc_lw)

        # NOTE: using Pprev here increases Poisson solver convergence speed a lot!
        P,is_convergent = compute_P(u, v, dx, dt, rho, Pprev=P, w=w, atol=atol)

        # third step (P)
        dPdx = df1_2(P, dx, axis=0)
        dPdy = df1_2(P, dx, axis=1)

        u = u - dt / rho * dPdx
        v = v - dt / rho * dPdy
        if not is_convergent:
            break
        # apply BCs one more at the end?
        #u,v = set_boundary(u,v)
    return u,v,P,is_convergent

def dt_fluid_flow(dx, Fo=0.3):
    """
    The stability threshold is found to be dt = 4e-5 for N,M=(50,50) i.e. dx=4e-5.
    It scales reliably as the inverse of dx² across many different N€[25;200].
    We can therefore define a Fourier number Fo=0.375
    (or slightly smaller to be absolutely safe) to get dt as a function of dx.
    """
    dt = dx**2 / nu * Fo
    return dt

@jit(nopython=True)
def advance_fluid_flow_2(Nt, u, v, f, dt, w=None, atol=1e-4, P=None):
    """
    This is the same but only keeps the previous data
    Evolve the fluid velocities u,v, according the incompressible N.-V.-Eqs
    over Nt time steps of duration dt.
    Note: the first time this function is executed takes long since numba needs to compile a lot).
    Args:
    - f: Time integration function, e.g. advance_adv_diff_RK3
    - P: Initial pressure field. Defaults to zeros if not provided.
    """
    UVP = np.zeros((Nt+1,3,*v.shape)) #data cube
    UVP[0,0,:] = u
    UVP[0,1,:] = v
    if P is None:
        P = np.zeros_like(u)
        UVP[0,2,:] = P

    N, M = u.shape

    dx, dy, Ns_c, Nc_lw = set_resolution(N,M)

    if w is None:
        w = 2 / (1 + sin(pi/N)) # optimal SOR parameter in the symmatrical case NxN

    for n in range(Nt):

        u = f(u, dt, u, v, dx, dy, nu)
        v = f(v, dt, u, v, dx, dy, nu)
        u,v = set_boundary(u,v, Ns_c, Nc_lw)

        # NOTE: using Pprev here increases Poisson solver convergence speed a lot!
        P,is_convergent = compute_P(u, v, dx, dt, rho, Pprev=P, w=w, atol=atol)

        # third step (P)
        dPdx = df1_2(P, dx, axis=0)
        dPdy = df1_2(P, dx, axis=1)

        u = u - dt / rho * dPdx
        v = v - dt / rho * dPdy
        UVP[n+1,0,:] = u
        UVP[n+1,1,:] = v
        UVP[n+1,2,:] = P

        if not is_convergent:
            break
        # apply BCs one more at the end?
        #u,v = set_boundary(u,v)
    return UVP,is_convergent
