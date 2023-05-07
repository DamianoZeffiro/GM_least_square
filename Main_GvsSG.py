import numpy as np
from RGM_ls import RGM_ls
from STGM_ls import STGM_ls
from GM_ls import GM_ls
from getData import getData
import matplotlib.pyplot as plt

# Optimality tolerance
eps = 1.0e-3
# Stopping criterion
stopcr = 1

# verbosity = 0 don't display info, verbosity = 1 display info
verb = 0

# Generation of the instance
n = 2**12
m = 2**10
k = round(0.1 * n)

Q, c, xs, xsn, picks = getData(m, n, k, 0, 0)

# starting point
x1 = np.zeros((n,))

fstop = 10**-9
maxit = 5000
maxit2 = 150000
maxit3 = 40000
lsg = 0.0001
lcgm = 10**7

print("*****************")
print("*  GM STANDARD  *")
print("*****************")

xgm, itergm, fxgm, tottimegm, fhgm, timeVecgm, gnrgm = GM_ls(Q, c, x1, lcgm, verb, 3, maxit, eps, fstop, stopcr)

# Print results
print("f(x)  = {:.3e}".format(fxgm))
print("Number of non-zero components of x = {}".format(np.sum(np.abs(xgm) >= 0.000001)))
print("Number of iterations = {}".format(itergm))
print("||gr||^2 = {}".format(gnrgm[-1]))
print("CPU time so far = {:.3e}".format(tottimegm))

print("*****************")
print("*  SGM STANDARD *")
print("*****************")

xagm, iteragm, fxagm, tottimeagm, fhagm, timeVecagm, gnragm = STGM_ls(Q, c, x1, lsg, verb, maxit2, eps, fstop, stopcr)

# Print results
print("f(x)  = {:.3e}".format(fxagm))
print("Number of iterations = {}".format(iteragm))
print("CPU time so far = {:.3e}".format(tottimeagm))

print("*****************")
print("*  RGM STANDARD *")
print("*****************")

xrgm, iterrgm, fxrgm, tottimergm, fhrgm, timeVecrgm, gnrrgm = RGM_ls(Q, c, x1, verb, 2, maxit3, eps, fstop, stopcr)

# Print results
print("f(x)  = {:.3e}".format(fxrgm))
print("Number of iterations = {}".format(iterrgm))
print("CPU time so far = {:.3e}".format(tottimergm))

# plot figure
fmin = 0.0

plt.semilogy(timeVecgm, fhgm - fmin, 'r-', label='GM')
plt.semilogy(timeVecagm, fhagm - fmin, 'b-', label='SGM')
plt.semilogy(timeVecrgm, fhrgm - fmin, 'y-', label='RGM')

plt.title('SGDvsGDvsRGM - objective function')
plt.legend()
plt.xlabel('time')
plt.ylabel('err')
plt.show()

