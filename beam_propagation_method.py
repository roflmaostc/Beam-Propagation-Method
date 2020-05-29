'''Homework 2, Computational Photonics, SS 2020:  Beam propagation method.
'''
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as lsp


def waveguide(xa, xb, Nx, n_cladding, n_core):
    '''Generates the refractive index distribution of a slab waveguide
    with step profile centered around the origin of the coordinate
    system with a refractive index of n_core in the waveguide region
    and n_cladding in the surrounding cladding area.
    All lengths have to be specified in µm.

    Parameters
    ----------
        xa : float
            Width of calculation window
        xb : float
            Width of waveguide
        Nx : int
            Number of grid points
        n_cladding : float
            Refractive index of cladding
        n_core : float
            Refractive index of core

    Returns
    -------
        n : 1d-array
            Generated refractive index distribution
        x : 1d-array
            Generated coordinate vector
    '''
    # generate coordinates
    x = np.linspace(-xa / 2, xa / 2, Nx)

    # generate the output array for the refractive index
    n_out = np.zeros(Nx)
    # fill cladding refractive index for all x larger than the
    # width of the waveguide
    n_out[np.abs(x) > xb // 2] = n_cladding
    # opposite for core
    n_out[np.abs(x) <= xb // 2] = n_core

    return n_out, x


def gauss(xa, Nx, w):
    '''Generates a Gaussian field distribution v = exp(-x^2/w^2) centered
    around the origin of the coordinate system and having a width of w.
    All lengths have to be specified in µm.

    Parameters
    ----------
        xa : float
            Width of calculation window
        Nx : int
            Number of grid points
        w  : float
            Width of Gaussian field

    Returns
    -------
        v : 1d-array
            Generated field distribution
        x : 1d-array
            Generated coordinate vector
    '''
    # generate coordinates
    x = np.linspace(-xa / 2, xa / 2, Nx)
    v = np.exp(- x ** 2 / w ** 2)
    return v, x


def beamprop_CN(v_in, lam, dx, n, nd,  z_end, dz, output_step):
    '''Propagates an initial field over a given distance based on the
    solution of the paraxial wave equation in an inhomogeneous
    refractive index distribution using the explicit-implicit
    Crank-Nicolson scheme. All lengths have to be specified in µm.

    Parameters
    ----------
        v_in : 1d-array
            Initial field
        lam : float
            Wavelength
        dx : float
            Transverse step size
        n : 1d-array
            Refractive index distribution
        nd : float
            Reference refractive index
        z_end : float
            Propagation distance
        dz : float
            Step size in propagation direction
        output_step : int
            Number of steps between field outputs

    Returns
    -------
        v_out : 2d-array
            Propagated field
        z : 1d-array
            z-coordinates of field output
    '''
    # k vector in vacuum
    k0 = 2 * np.pi / lam
    # expected average k vector
    kbar = k0 * nd

    # array containing the output positions
    N_iter = round(z_end / dz)
    N_stor = round(z_end / dz / output_step)

    v_out = np.zeros((N_stor, len(v_in)), dtype=np.complex64)
    z = np.arange(0, z_end, dz * output_step)

    # matrix L construction
    # first we create the diagonal matrix
    Aj = dz / 2 * (- 1j / (kbar * dx * dx)
                   + 1j * (k0 * k0 * n * n - kbar * kbar)/(2 * kbar))
    # the off diagonals
    Bj = 1j / (2 * kbar * dx * dx)
    # construct a sparse matrix in the csr format
    Mr = sps.diags([1 + Aj, Bj, Bj], [0, -1, 1], format="csr")
    Ml = sps.diags([1 - Aj, - Bj, - Bj], [0, -1, 1], format="csr")

    for i in range(N_iter):
        # save every output_step the intermediate results in v_out
        if i % output_step == 0:
            v_out[i // output_step, :] = v_in

        # calculate the next step with solving the linear system of equations
        v_in = lsp.spsolve(Ml, Mr.dot(v_in))

    return v_out, z
