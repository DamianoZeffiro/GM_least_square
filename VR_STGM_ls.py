import numpy as np
import time


def VR_STGM_ls(Q, c, x, lc, verbosity, nepochs, maxit, eps, fstop, stopcr):
    """
    Implementation of the Stochastic Variance Reduction
    Gradient Method
    for min f(x)=0.5 ||Qx-c||^2

    INPUTS:
    Q: Q matrix
    c: c term
    x: starting point
    lc: constant of the reduced stepsize (numerator)
    verbosity: printing level
    nepochs: epoch length
    maxit: maximum number of iterations
    eps: tolerance
    fstop: target o.f. value
    stopcr: stopping condition
    """

    maxniter = maxit
    fh = np.zeros(maxit)
    gnrit = np.zeros(maxit)
    timeVec = np.zeros(maxit)
    flagls = 0

    start_time = time.time()
    timeVec[0] = 0

    m, n = Q.shape

    # Values for the smart computation of the o.f.
    rx = Q @ x - c
    rzsvrg = rx
    gsvrg = Q.T @ rzsvrg
    fx = 0.5 * np.sum(rx ** 2)

    it = 1

    while flagls == 0:
        # Vectors updating
        if it == 1:
            timeVec[it - 1] = 0
        else:
            timeVec[it - 1] = time.time() - start_time
        fh[it - 1] = fx

        # Gradient evaluation
        ind = np.random.randint(m)
        g = (Q[ind, :] @ x - c[ind]) * Q[ind, :].T
        gz = rzsvrg[ind] * Q[ind, :].T
        gf = g - gz + (1 / m) * gsvrg
        d = -gf

        gnr = g @ d
        gnrit[it - 1] = -gnr

        # Stopping criteria and test for termination
        if it >= maxniter:
            break
        if stopcr == 1:
            if fx <= fstop:
                break
        elif stopcr == 2:
            if abs(gnr) <= eps:
                break
        else:
            raise ValueError("Unknown stopping criterion")

        # Alpha selection
        alpha = lc
        z = x + alpha * d
        # Update gradient at the end of epoch (10000 iterations per epoch)
        if it % nepochs == 0:
            rz = Q @ z - c
            fz = 0.5 * np.sum(rz ** 2)
            rzsvrg = rz
            gsvrg = Q.T @ rz
        else:
            rz = rx
            fz = fx

        x = z
        fx = fz
        rx = rz

        if verbosity > 0:
            print(f"-----------------** {it} **------------------")
            print(f"gnr      = {abs(gnr)}")
            print(f"f(x)     = {fx}")
            print(f"alpha    = {alpha}")

        it += 1

    if it < maxit:
        fh[it - 1:maxit] = fh[it - 2]
        gnrit[it - 1:maxit] = gnrit[it - 2]
        timeVec[it - 1:maxit] = timeVec[it - 2]

    ttot = time.time() - start_time

    return x, it, fx, ttot, fh, timeVec, gnrit
