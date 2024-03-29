from numba import jit
from scipy.constants import N_A ,m_p
from numpy import exp
import numpy as np
from Funcs import advance_adv_diff_RK3, advance_adv_diff_RK4
from parameters import *
from pathlib import Path
from fluid_flow import dt_fluid_flow

species_names = ['CH$_4$', 'O$_2$', 'N$_2$',  'H$_2$O', 'CO$_2$']
species_names_2 = ['CH_4', 'O_2', 'N_2',  'H_2O', 'CO_2']

mH = 1.0079
mO = 16.000
mC = 12.0107
mN = 14.0067

# molar masses
W = np.array([mC + 4*mH, 2*mO, 2*mN, 2*mH+mO, mC+2*mO]) * 1e-3 # kg/mol

# enthalpy of formation
dh0 = np.array([-74.9, 0, 0, -241.818, -393.52]) * 1e3 # J/mol

# stochiomteric coefficients
nu_stoch = np.array([-1, -2, 0, 2, 1], dtype=float)

# group the boundary conditions (respecting the chosen species ordering)
@jit(nopython=True)
def set_BCs(Y, Ns_c, Nc_lw):
    Y[0,:] = set_CH4_BC(Y[0,:], Ns_c, Nc_lw)
    Y[1,:] = set_O2_BC(Y[1,:],  Ns_c, Nc_lw)
    Y[2,:] = set_N2_BC(Y[2,:],  Ns_c, Nc_lw)
    Y[3,:] = set_H2O_BC(Y[3,:], Ns_c, Nc_lw)
    Y[4,:] = set_CO2_BC(Y[4,:], Ns_c, Nc_lw)
    return  Y

@jit(nopython=True)
def set_CH4_BC(Y_CH4, Ns_c, Nc_lw):

    # set to Neumann BC first everywhere:
    Y_CH4[0,:]  = Y_CH4[1,:]
    Y_CH4[-1,:] = Y_CH4[-2,:]
    Y_CH4[:,0]  = Y_CH4[:,1]
    Y_CH4[:,-1] = Y_CH4[:,-2]

    #taking the gas out of O2 inlets
    Y_CH4[:Ns_c, 0] = 0
    #taking the gas out of N2 inlets
    for j in [0,-1]:
        Y_CH4[Ns_c:Nc_lw, j] = 0
    # inlet (upper left corner)
    Y_CH4[:Ns_c, -1] = 1

    return Y_CH4

@jit(nopython=True)
def set_O2_BC(Y_O2, Ns_c, Nc_lw):

    # set to Neumann BC first everywhere:
    Y_O2[0,:]  = Y_O2[1,:]
    Y_O2[-1,:] = Y_O2[-2,:]
    Y_O2[:,0]  = Y_O2[:,1]
    Y_O2[:,-1] = Y_O2[:,-2]

    #taking the gas out of CH4 inlets
    Y_O2[:Ns_c, -1] = 0
    #taking the gas out of N2 inlets
    for j in [0,-1]:
        Y_O2[Ns_c:Nc_lw, j] = 0
    # inlet (lower left corner)
    Y_O2[:Ns_c, 0] = .233

    return Y_O2

@jit(nopython=True)
def set_N2_BC(Y_N2, Ns_c, Nc_lw):

    # set to Neumann BC first everywhere:
    Y_N2[0,:]  = Y_N2[1,:]
    Y_N2[-1,:] = Y_N2[-2,:]
    Y_N2[:,0]  = Y_N2[:,1]
    Y_N2[:,-1] = Y_N2[:,-2]

    #taking the gas out of CH4 inlets
    Y_N2[:Ns_c, -1] = 0
    # inlet (lower and upper coflow inlets)
    for j in [0,-1]:
        Y_N2[Ns_c:Nc_lw, j] = 1

    Y_N2[:Ns_c, 0] = .767


    return Y_N2

@jit(nopython=True)
def set_CO2_BC(Y_CO2, Ns_c, Nc_lw):

    # set to Neumann BC first everywhere:
    Y_CO2[0,:]  = Y_CO2[1,:]
    Y_CO2[-1,:] = Y_CO2[-2,:]
    Y_CO2[:,0]  = Y_CO2[:,1]
    Y_CO2[:,-1] = Y_CO2[:,-2]
    #taking the gas out of CH4 inlets
    Y_CO2[:Ns_c, -1] = 0
    #taking the gas out of O2 inlets
    Y_CO2[:Ns_c, 0] = 0
    #taking the gas out of N2 inlets
    for j in [0,-1]:
        Y_CO2[Ns_c:Nc_lw, j] = 0

    return Y_CO2

@jit(nopython=True)
def set_H2O_BC(Y_H2O, Ns_c, Nc_lw):

    # set to Neumann BC first everywhere:
    Y_H2O[0,:]  = Y_H2O[1,:]
    Y_H2O[-1,:] = Y_H2O[-2,:]
    Y_H2O[:,0]  = Y_H2O[:,1]
    Y_H2O[:,-1] = Y_H2O[:,-2]
    #taking the gas out of CH4 inlets
    Y_H2O[:Ns_c, -1] = 0
    #taking the gas out of O2 inlets
    Y_H2O[:Ns_c, 0] = 0
    #taking the gas out of N2 inlets
    for j in [0,-1]:
        Y_H2O[Ns_c:Nc_lw, j] = 0

    return Y_H2O

@jit(nopython=True)
def set_Temp_BC(Temp, Ns_c, Nc_lw):

    # set to Neumann BC first everywhere:
    Temp[0,:]  = Temp[1,:]
    Temp[-1,:] = Temp[-2,:]
    Temp[:,0]  = Temp[:,1]
    Temp[:,-1] = Temp[:,-2]

    # inlets (lower and upper slot + coflow)
    for j in [0,-1]:
        Temp[:Nc_lw, j] = 300 # Kelvin

    return Temp

# group the boundary conditions (respecting the chosen species ordering)
BCs = [set_CH4_BC, set_O2_BC, set_N2_BC, set_H2O_BC, set_CO2_BC]


@jit(nopython=True)
def Y_to_n(Y):
    """Converts mass fraction Y to volumic density n in mol/m⁻³."""
    n = np.zeros_like((Y))
    nspec = Y.shape[0]
    for k in range(nspec):
        n[k] = Y[k] * rho / W[k]

    return n

@jit(nopython=True)
def get_Q(n_CH4, n_O2, T, TA=1e4):
    """
    Express the densities of CH4 and O2 in m⁻³, Temperature T in Kelvin.
    """
    A = 1.1e8
    return A * n_CH4 * n_O2**2 * exp(- TA / T)

@jit(nopython=True)
def advance_chem(Y,T,dt_chem):
    n = Y_to_n(Y)
    Q = get_Q(n[0], n[1], T)
    nspec = Y.shape[0]
    omega_dot = np.zeros(nspec)
    for k in range(nspec):
        omega_dot[k] = W[k] * nu_stoch[k] * Q
        Y[k] += dt_chem * omega_dot[k] / rho

    omegaT_dot = - np.sum(dh0 / W * omega_dot)
    T += dt_chem * omegaT_dot / rho / cp
    return Y,T

@jit(nopython=True)
def integr_chem_0d(Y, T, dt, Nt_chem):
    """
    Integration of the chemistry equations for the homogeneous reactor.
    Args:
        - Y:  mass fraction of species k
        - T:  Temperature in K
        - dt: integrates the chemical equation from t to t+dt
        - Nt_chem: number of time steps for the integration (dt_chem = dt/Nt_chem)
    """
    dt_chem = dt / Nt_chem

    T_t = np.zeros(Nt_chem)
    for n in range(Nt_chem):
        T_t[n] = T
        Y,T = advance_chem(Y,T,dt_chem)
    return Y, T_t

@jit(nopython=True)
def integr_chem_2d(Y, T, dt, dt_chem, evolve_T=True):
    """
    Integration of the chemistry equations for the whole chamber.
    Args:
        - Y:  mass fraction of species k
        - T:  Temperature in K
        - dt: integrates the chemical equation from t to t+dt
        - dt_chem: time step for the integration (dt_chem = dt/Nt_chem)
    """
    Nt_chem = dt // dt_chem

    for n in range(Nt_chem):
        n = Y_to_n(Y)
        Q = get_Q(n[0], n[1], T)
        omega_dot = np.zeros_like(Y)
        nspec = Y.shape[0]
        for k in range(nspec):
            omega_dot[k] = W[k] * nu_stoch[k] * Q
            Y[k] += dt_chem * omega_dot[k] / rho

        if evolve_T:
            omegaT_dot = np.zeros_like(T)
            for k in range(nspec):
                omegaT_dot -= dh0[k] / W[k] * omega_dot[k]
            T += dt_chem * omegaT_dot / rho / cp

    return Y, T


@jit(nopython=True)
def evolve_species(Nt, Y, T, dt, u, v, dx, dy, Ns_c, Nc_lw, chem=True, dt_chem=None, evolve_T=True):

    nspec = Y.shape[0]
    Y = set_BCs(Y, Ns_c, Nc_lw)
    T = set_Temp_BC(T, Ns_c, Nc_lw)

    if dt_chem is None:
        dt_chem = dt / 100

    for n in range(Nt):
        if chem:
            Y, _ = integr_chem_2d(Y, T, dt, dt_chem, evolve_T)
        if evolve_T:
            T = _
            # (almost) same procedure for temperature as for species:
            T = advance_adv_diff_RK3(T, dt, u, v, dx, dy, nu)
            T = set_Temp_BC(T, Ns_c, Nc_lw)

        for k in range(nspec):

            if k < nspec-1:
                Y[k] = advance_adv_diff_RK3(Y[k], dt, u, v, dx, dy, nu)
            else:
                # normalization condition:
                Y[k] = 1 + Y[k] - np.sum(Y, axis=0)
        Y = set_BCs(Y, Ns_c, Nc_lw)

    return Y, T

def set_up_T(N,M, dy, Tcent=1000, T0=300, d=0.5e-3, smooth=False):
    """Set up the fixed high temperature in the center to initiate the combustion"""
    from Funcs import tanh_transition
    if smooth:
        T = tanh_transition(np.arange(M), 0, 1, M/2-d/dy, d/dy/4)
        T *= tanh_transition(np.arange(M), 1, 0, M/2+d/dy, d/dy/4)
    else:
        T = np.zeros(M)
        T[int(M/2-d/dy) : int(M/2+d/dy) + 1] = 1
    T = (Tcent - T0) * T + T0
    T = np.reshape(T, (1, -1))
    T = T.repeat(N, axis=0)
    return T


def compute_Y_pre_combustion(N, u, v, t=0.04):
    
    dx, dy, Ns_c, Nc_lw = set_resolution(N,N)
    dt = dt_fluid_flow(dx, Fo=0.3)

    # number of iterations
    Nt = int(t/dt)
    
    # initial setup of the species distribution
    CH4, O2, N2, H2O, CO2, T = np.zeros((6,N,N))
    O2[:] = .233
    N2[:] = .767
    T[:] = 300
    Y = np.array([CH4, O2, N2, CO2, H2O])
    
    # iterate:
    Y, T = evolve_species(Nt, Y, T, dt, u, v, dx, dy, Ns_c, Nc_lw, chem=False, evolve_T=False)
    
    return Y,T

def compute_Y_combustion(N, u, v, Y, T, t=0.03):
    """
    Combustion step at fixed Temperature.
    """
    
    dx, dy, Ns_c, Nc_lw = set_resolution(N,N)
    dt = dt_fluid_flow(dx, Fo=0.3)

    # number of iterations
    Nt = int(t/dt)
    
    # iterate:
    Y, T = evolve_species(Nt, Y, T, dt, u, v, dx, dy, Ns_c, Nc_lw, chem=True, dt_chem=6.4e-7, evolve_T=False)
    
    return Y,T

def compute_Y_combustion_with_T(N, u, v, Y, T, t=0.03):
    """
    Final combustion step with evolution of T.
    """
    
    dx, dy, Ns_c, Nc_lw = set_resolution(N,N)
    dt = dt_fluid_flow(dx, Fo=0.3)
    dt_chem = get_dt_chem(N)

    # number of iterations
    Nt = int(t/dt)
    
    # iterate:
    Y, T = evolve_species(Nt, Y, T, dt, u, v, dx, dy, Ns_c, Nc_lw, chem=True, dt_chem=dt_chem, evolve_T=True)
    
    return Y,T


def save_Y_T(Y,T,p):
    species_data = np.zeros((Y.shape[0] + 1, *Y.shape[1:]))
    species_data[:-1,:, :] = Y
    species_data[-1, :, :] = T
    np.save(p, species_data)

def load_Y_T(p):
    species_data = np.load(p)
    Y = species_data[:-1,:, :]
    T = species_data[-1, :, :]
    return Y,T

@jit(nopython=True)
def get_dt_chem(N):
    """
    Returns time step in [s] for chemistry integration (with T evolution), obtained from
    a linear fit at stability threshold (tested for evolve_species with evolve_T=True
    and it works for both RK3 and RK4). 
    There are exceptions from the linear behavior
        - at N=50, no explanation yet (workaround: reduce dt_chem by factor 0.8)
        - beyond N=130; it seems that dt_chem cannot be higher than ~dt/6 (which corresponds to N=130)
    """
    dt_chem = 5.121e-9 * N - 1.115e-7
    
    
    dx, dy, Ns_c, Nc_lw = set_resolution(N,N)
    dt = dt_fluid_flow(dx, Fo=0.3)
    
    if dt_chem > dt/6:
        return dt/6
    else:
    
        if N>30 and N<70:
            return 0.8 * dt_chem
        else:
            return dt_chem
