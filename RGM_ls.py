import numpy as np
import time

# slower than matlab version, perhaps can be optimized

def RGM_ls(Q, c, x, verbosity, arls, maxit, eps, fstop, stopcr):
    """
    Implementation of the Randomized BCGD Method
    for min f(x)=0.5 * ||Qx-c||^2

    INPUTS:
    Q: Q matrix
    c: c term
    x: starting point
    verbosity: printing level
    arls: line search (1 Armijo 2 exact)
    maxit: maximum number of iterations
    eps: tolerance
    fstop: target o.f. value
    stopcr: stopping condition
    """

    gamma = 0.0001
    maxniter = maxit
    fh = np.zeros(maxit)
    gnrit = np.zeros(maxit)
    timeVec = np.zeros(maxit)
    flagls = 0

    m, n = Q.shape

    start_time = time.time()
    timeVec[0] = 0

    # Values for the computation of the o.f.
    # Values for the smart computation of the o.f.
    rx = Q @ x - c
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
        while True:
            ind = np.random.randint(n)
            Qix = Q[:, ind].T @ rx
            gi = Qix
            Qii = np.linalg.norm(Q[:, ind]) ** 2
            d = -gi
            if d != 0.0:
                break

        gnr = gi * d
        gnrit[it - 1] = gnr

        # Stopping criteria and test for termination
        if it >= maxniter:
            break

        if stopcr == 1:
            # Continue if not yet reached target value fstop
            if fx <= fstop:
                break
        elif stopcr == 2:
            # Stopping criterion based on the product of the
            # gradient with the direction
            if abs(n * gnr) <= eps:
                break
        else:
            raise ValueError("Unknown stopping criterion")

        # Set z=x
        z = x.copy()

        # Line search
        if arls == 1:
            # Armijo search
            alpha = 1.0
            ref = gamma * gnr

            while True:
                z[ind] = x[ind] + alpha * d
                # Smart computation of the o.f. at the trial point
                fz = fx + alpha * d * gi + 0.5 * (alpha * d) ** 2 * Qii

                if fz <= fx + alpha * ref:
                    z[ind] = x[ind] + alpha * d
                    break
                else:
                    alpha = alpha * 0.1

                if alpha <= 1e-20:
                    z = x
                    fz = fx
                    flagls = 1
                    it = it - 1
                    break
        else:
            # Exact alpha
            alpha = 1 / Qii
            z[ind] = x[ind] + alpha * d
            fz = fx + alpha * d * gi + 0.5 * (alpha * d)**2 * Qii

        # Update x, residual, and fx
        x = z
        rx = rx - alpha * Q[:, ind] * (Q[:, ind].T @ rx)
        fx = fz
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
