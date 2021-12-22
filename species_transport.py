from numba import jit
from scipy.constants import N_A ,m_p
from numpy import exp
import numpy as np

from parameters import *

species_names = ['CH$_4$', 'O$_2$', 'N$_2$',  'H$_2$O', 'CO$_2$']

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
def integr_chem_2d(Y, T, dt, Nt_chem, evolve_T=True):
    """
    Integration of the chemistry equations for the whole chamber.
    Args:
        - Y:  mass fraction of species k
        - T:  Temperature in K
        - dt: integrates the chemical equation from t to t+dt
        - Nt_chem: number of time steps for the integration (dt_chem = dt/Nt_chem)
    """
    dt_chem = dt / Nt_chem

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
