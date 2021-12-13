from numba import jit
from parameters import *

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
