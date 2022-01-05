import numpy as np
from numba import jit

@jit(nopython=True)
def SOR_solver(b, Pprev=None, w=1, atol=1e-4, maxit=1000000):
    """
    Solve the elliptic pressure equation (formally identical to te Poisson eq.):
    ΔP = b (Δ is normalized 2D Laplace operator discretized using centered
    finite differences. Note that (dx)² is included in b.
    The algorithm usese the SOR method seen in the lectures.
    The BCs are chosen such that P=0 at the right boundary and V.Neumann
    conditions an all other boundaries.

    """

    N,M = b.shape
    is_convergent=True

    # if the pressure field at the previous iteration is not too different,
    # the algorithm might converge faster
    if Pprev is None:
        Pact = np.zeros((N,M)) # intial guess
    else:
        Pact = np.copy(Pprev)

    # the convergence condition (maybe another metric is better/more efficient)
    def converged(V0, V1):
        return np.mean(np.abs(V0-V1)) < atol

    def dP_BC(i,j):
        """
        Returns the quantites dP and b taking into account boundary conditions
        """
        _b = 0
        dP = 0

        # to test the 2D Poisson situation we used these BCs:
        # if i==N-1:
        #     dP = 4 * 45
        # elif i==0:
        #     dP = 4 * 10

        #in this problem, we have:
        #P=0 at right border
        if i==N-1:
            dP = 0

        # Neumann at other borders:
        elif i==0:
            dP = 4 * Pact[i+1,j]
        elif j==0:
            dP = 4 * Pact[i,j+1]
        elif j==M-1:
            dP = 4 * Pact[i,j-1]
        else:
            dP = Pact[i-1,j] + Pact[i,j-1] + Pact[i+1,j] + Pact[i,j+1]
            _b = b[i,j]
        return dP, _b

    n = 0
    while True:
        Pold = np.copy(Pact) # needed to check the convergence

        """arguments: w between 0 and 2
        Return x a vector of the same size of a"""

        "Verifying that the inputs are correct"
        assert w > 0 and w < 2 , "the argument is not between 0 and 2"

        for i in range(N):
            for j in range(M):
                dP, _b = dP_BC(i,j)
                Pact[i,j] = (1-w) * Pact[i,j] + w * (1/4 * dP - 1/4 * _b)

        if converged(Pold, Pact):
            break
        if n > maxit:
            print('did not converge within the maximum number of iterations')
            is_convergent = False
            break
        n += 1
    return Pact,is_convergent