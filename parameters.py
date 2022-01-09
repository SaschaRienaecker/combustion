from numba import jit


# all units are in SI
# N       = 50
# M       = 50
Lx      = 0.002  # m
Ly      = 0.002  # m
Lslot   = 0.0005 # m
Lcoflow = 0.0005 # m
# dt      = 1e-7   # s
nu      = 15e-6  # m²s⁻¹
#nu      = 10**-3 # m²s⁻¹
rho     = 1.1614 # kg m⁻³
cp      = 1200 # J / kg / K
# dx      = Lx/N
# dy      = Ly/N
# Ns_c    = int(Lslot          /dx) #N for the point between the slot and te coflow
# Nc_lw   = int((Lslot+Lcoflow)/dx) #N for the point between the coflow and the rest of the wall

Uslot   = 1  # inlet  speed in m/s
Ucoflow = .2 # coflow speed in m/s

@jit(nopython=True)
def set_resolution(N, M):
    dx      = Lx/(N-1)
    dy      = Ly/(M-1)
    Ns_c    = int(Lslot          /dx) #N for the point between the slot and te coflow
    Nc_lw   = int((Lslot+Lcoflow)/dx) #N for the point between the coflow and the rest of the wall
    return dx, dy, Ns_c, Nc_lw
