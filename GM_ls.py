import numpy as np
import time


def GM_ls(Q, c, x, lc, verbosity, arls, maxit, eps, fstop, stopcr):
    """
    Implementation of the Gradient Method for minimizing f(x) = 0.5 * ||Qx - c||^2

    INPUTS:
    Q : Q matrix
    c : c term
    x : starting point
    lc : Lipschitz constant of the gradient (not needed if exact/Armijo ls used)
    verbosity : printing level
    arls : line search (1 Armijo 2 exact 3 fixed)
    maxit : maximum number of iterations
    eps : tolerance
    fstop : target o.f. value
    stopcr : stopping condition
    """
    gamma = 0.0001
    maxniter = maxit
    fh = np.zeros(maxit)
    gnrit = np.zeros(maxit)
    timeVec = np.zeros(maxit)
    flagls = 0

    start_time = time.time()
    timeVec[0] = 0

    # Values for the smart computation of the o.f.
    rx = Q @ x - c

    fx = 0.5 * np.dot(rx, rx)

    it = 1

    while flagls == 0:
        if it == 1:
            timeVec[it - 1] = 0
        else:
            timeVec[it - 1] = time.time() - start_time

        fh[it - 1] = fx

        # Gradient evaluation
        g = Q.T @ rx
        d = -g

        gnr = np.dot(g, d)
        gnrit[it - 1] = -gnr

        # Stopping criteria and test for termination
        if it >= maxniter:
            break

        if stopcr == 1:
            # Continue if not yet reached target value fstop
            if fx <= fstop:
                break
        elif stopcr == 2:
            # Stopping criterion based on the product of the gradient with the direction
            if abs(gnr) <= eps:
                break
        else:
            raise ValueError('Unknown stopping criterion')

        # Line search
        if arls == 1:
            # Armijo search
            alpha = 1
            ref = gamma * gnr

            while True:
                z = x + alpha * d
                rz = Q @ z - c
                fz = 0.5 * np.dot(rz, rz)

                if fz <= fx + alpha * ref:
                    z = x + alpha * d
                    break
                else:
                    alpha *= 0.1

                if alpha <= 1e-20:
                    z = x
                    fz = fx
                    flagls = 1
                    it -= 1
                    break

        elif arls == 2:
            # Exact alpha
            den = np.linalg.norm(Q @ d, 2) ** 2
            alpha = -gnr / den
            z = x + alpha * d
            rz = Q @ z - c
            fz = 0.5 * np.dot(rz, rz)

        else:
            # Fixed alpha
            alpha = 1 / lc
            z = x + alpha * d
            rz = Q @ z - c
            fz = 0.5 * np.dot(rz, rz)

        x = z
        rx = rz
        fx = fz

        if verbosity > 0:
            print(f'-----------------** {it} **------------------')
            print(f'gnr      = {abs(gnr)}')
            print(f'f(x)     = {fx}')
            print(f'alpha    = {alpha}')

        it += 1

    if it < maxit:
        fh[it - 1:maxit] = fh[it - 2]
        gnrit[it - 1:maxit] = gnrit[it - 2]
        timeVec[it - 1:maxit] = timeVec[it - 2]

    ttot = time.time() - start_time

    return x, it, fx, ttot, fh, timeVec, gnrit
