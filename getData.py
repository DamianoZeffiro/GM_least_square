import numpy as np
from scipy.linalg import hadamard

"""
COMPRESSED SENSING PROBLEMS

GENERAL DESCRIPTION AND OUTPUTS

function [A, b, xs, xsn, picks] = getData(m, n, k, Ameth, xmeth, varargin)

Generates data for compressed sensing problems.

A      - m x n matrix if A is explicit; empty otherwise
b      - m x 1 vector equal to A*(xs+noise) + noise
xs     - n x 1 vector with k nonzeros
xsn    - n x 1 vector xs + noise
picks  - vector of row indices if A is implicit; empty otherwise

For compressed sensing, m <= n, and if k is small enough l1 regularized
least squares applied to A and b will recover xs.

INPUTS

Ameth determines how A is generated
  0 - randn(m,n)
  1 - randn then columns scaled to unit norm
  2 - randn then QR to get orthonormal rows
  3 - bernoulli +/- 1 distribution
  4 - partial Hadamard matrix
  5 - picks for partial fourier matrix
  6 - picks for partial discrete cosine transform matrix
      REQUIRES Signal Processing Toolbox
  7 - picks for partial 2-d discrete cosing transform matrix
      REQUIRES Image Processing Toolbox

xmeth determines how xs is generated
  0 - randperm for support, 2*randn for values
  1 - randperm for support, 2*(rand - 0.5) for values
  2 - randperm for support, ones for values
  3 - randperm for support, sign(randn) for values

varargin{1} = sigma1 - standard deviation of signal noise (added to xs)
varargin{2} = sigma2 - standard deviation of meas. noise (added to b)
varargin{3} = state - state used to initialize rand and randn (scalar
                       integer 0 to 2^32-1)
"""

def getData(m, n, k, Ameth, xmeth, sigma1=None, sigma2=None, state=None):
    if state is not None:
        np.random.seed(state)

    picks = []

    # Generate matrix A based on Ameth
    if Ameth == 0:
        # randn, no scaling
        A = np.random.randn(m, n)
    elif Ameth == 1:
        # randn, column scaling
        A = np.random.randn(m, n)
        for i in range(n):
            A[:, i] /= np.linalg.norm(A[:, i])
    elif Ameth == 2:
        # randn, orthonormal rows
        A = np.random.randn(m, n)
        A, _ = np.linalg.qr(A.T, mode='economic')
        A = A.T
    elif Ameth == 3:
        # bernoulli +/- 1
        A = np.sign(np.random.rand(m, n) - 0.5)
        ind = np.where(A == 0)
        A[ind] = 1
    elif Ameth == 4:
        # partial hadamard
        A = hadamard(n)
        picks = np.random.permutation(n)[:m]
        A = A[picks, :]
    elif Ameth == 5:
        # partial 1-d fourier transform
        A = None  # You will need to implement or import a partial Fourier matrix function
        picks = np.random.permutation(n)[:m]
    elif Ameth == 6:
        # partial 1-d discrete cosine transform
        A = None  # You will need to implement or import a partial DCT matrix function
        picks = np.random.permutation(n)[:m]
    elif Ameth == 7:
        # partial 2-d discrete cosine transform
        A = None  # You will need to implement or import a partial 2D DCT matrix function
        picks = np.random.permutation(n)[:m]

    # Generate xs based on xmeth
    p = np.random.permutation(n)
    xs = np.zeros(n)
    if xmeth == 0:
        xs[p[:k]] = 2 * np.random.randn(k)
    elif xmeth == 1:
        xs[p[:k]] = 2 * (np.random.rand(k) - 0.5)
    elif xmeth == 2:
        xs[p[:k]] = np.ones(k)
    elif xmeth == 3:
        xs[p[:k]] = np.sign(np.random.randn(k))
    # ... other cases for xmeth

    # Add noise to xs and xsn
    xsn = xs.copy()
    if sigma1 is not None and sigma1 > 0:
        xsn += np.random.randn(n) * sigma1

    # Get noiseless measurements
    b = np.dot(A, xsn)

    # Add noise to b
    if sigma2 is not None and sigma2 > 0:
        b += np.random.randn(m) * sigma2

    return A, b, xs, xsn, picks
