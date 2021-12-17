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


@jit(nopython=True)
def set_CH4_BC(Y_CH4):

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
def set_O2_BC(Y_O2):

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
def set_N2_BC(Y_N2):

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
def set_CO2_BC(Y_CO2):

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
def set_H2O_BC(Y_H2O):

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
def set_Temp_BC(Temp):

    # set to Neumann BC first everywhere:
    Temp[0,:]  = Temp[1,:]
    Temp[-1,:] = Temp[-2,:]
    Temp[:,0]  = Temp[:,1]
    Temp[:,-1] = Temp[:,-2]

    # inlets (lower and upper slot + coflow)
    for j in [0,-1]:
        Temp[:Nc_lw, j] = 300 # Kelvin

    return Temp

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