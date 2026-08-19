import numpy as np
from scipy.integrate import quad_vec
from math import sqrt, exp, inf, log1p, pi
from scipy.constants import c

def k_integrand_energy(k, k0, d, epsm, rL, rR, mum=1.):
    """
    Integrand of the radial part of in-plane wave vector for the Casimir energy.

    Parameters
    ----------
    k : float
        in-plane wave vector
    k0 : float
        vacuum wave number
    d : float
        separation between the two plates
    epsm : float
        dielectric function of the medium evaluated at the vacuum wave number
    rL, rR : float
        reflection coefficient of the left and right plate, respectively
    mum : float
        magnetic permeability of the medium evaluated at the vacuum wave number

    Returns
    -------
    (float, float)
        result for TE and TM polarization
    """
    kappa = sqrt(epsm * mum * k0 ** 2 + k ** 2)
    rTM_L, rTE_L = rL(k0, k)
    rTM_R, rTE_R = rR(k0, k)
    res_TE = k / 2 / pi * log1p(- rTE_L * rTE_R * exp(-2 * kappa * d))
    res_TM = k / 2 / pi * log1p(- rTM_L * rTM_R * exp(-2 * kappa * d))
    return res_TE, res_TM

def _integrate_k0_contribution(integrand, k0, d, epsm_func, rL, rR, epsrel=1.e-8, epsabs=0.0, mum_func=None):
    epsm = epsm_func(k0 * c)
    mum = 1. if mum_func is None else mum_func(k0 * c)
    f = lambda t: np.array(integrand(t / d, k0, d, epsm, rL, rR, mum)) / d
    return quad_vec(f, 0, inf, epsrel=epsrel, epsabs=epsabs)[0]


def k0_func_energy(k0, d, epsm_func, rL, rR, epsrel=1.e-8, epsabs=0.0, mum_func=None):
    """
    Casimir free energy contribution at a given wave number k0.

    Parameters
    ----------
    k0 : float
        vacuum wave number
    d : float
        separation between the two plates
    epsm_func : function
        dielectric function of the medium
    rL, rR : float
        reflection coefficient of the left and right plate, respectively
    mum_func : function or None
        magnetic permeability of the medium. If None, the medium is treated as non-magnetic.

    Returns
    -------
    numpy array of two floats
        result for TE and TM polarization
    """
    return _integrate_k0_contribution(k_integrand_energy, k0, d, epsm_func, rL, rR, epsrel=epsrel, epsabs=epsabs, mum_func=mum_func)

def k_integrand_pressure(k, k0, d, epsm, rL, rR, mum=1.):
    """
    Integrand of the radial part of in-plane wave vector for the Casimir pressure.

    Parameters
    ----------
    k : float
        in-plane wave vector
    k0 : float
        vacuum wave number
    d : float
        separation between the two plates
    epsm : float
        dielectric function of the medium evaluated at the vacuum wave number
    rL, rR : float
        reflection coefficient of the left and right plate, respectively
    mum : float
        magnetic permeability of the medium evaluated at the vacuum wave number

    Returns
    -------
    (float, float)
        result for TE and TM polarization
    """
    kappa = sqrt(epsm * mum * k0 ** 2 + k ** 2)
    rTM_L, rTE_L = rL(k0, k)
    rTM_R, rTE_R = rR(k0, k)
    res_TE = -2 * k * kappa / 2 / pi * rTE_L * rTE_R * exp(-2 * kappa * d) / (1 - rTE_L * rTE_R * exp(-2 * kappa * d))
    res_TM = -2 * k * kappa / 2 / pi * rTM_L * rTM_R * exp(-2 * kappa * d) / (1 - rTM_L * rTM_R * exp(-2 * kappa * d))
    return res_TE, res_TM

def k0_func_pressure(k0, d, epsm_func, rL, rR, epsrel=1.e-8, epsabs=0.0, mum_func=None):
    """
    Casimir pressure contribution at a given wave number k0.

    Parameters
    ----------
    k0 : float
        vacuum wave number
    d : float
        separation between the two plates
    epsm_func : function
        dielectric function of the medium
    rL, rR : float
        reflection coefficient of the left and right plate, respectively
    mum_func : function or None
        magnetic permeability of the medium. If None, the medium is treated as non-magnetic.

    Returns
    -------
    numpy array of two floats
        result for TE and TM polarization
    """
    return _integrate_k0_contribution(k_integrand_pressure, k0, d, epsm_func, rL, rR, epsrel=epsrel, epsabs=epsabs, mum_func=mum_func)

def k_integrand_pressuregradient(k, k0, d, epsm, rL, rR, mum=1.):
    """
    Integrand of the radial part of in-plane wave vector for the Casimir pressure gradient.

    Parameters
    ----------
    k : float
        in-plane wave vector
    k0 : float
        vacuum wave number
    d : float
        separation between the two plates
    epsm : float
        dielectric function of the medium evaluated at the vacuum wave number
    rL, rR : float
        reflection coefficient of the left and right plate, respectively
    mum : float
        magnetic permeability of the medium evaluated at the vacuum wave number
    Returns
    -------
    (float, float)
        result for TE and TM polarization
    """
    kappa = sqrt(epsm * mum * k0 ** 2 + k ** 2)
    rTM_L, rTE_L = rL(k0, k)
    rTM_R, rTE_R = rR(k0, k)
    res_TE = 4 * k * kappa ** 2 / 2 / pi * rTE_L * rTE_R * exp(-2 * kappa * d) / (1 - rTE_L * rTE_R * exp(-2 * kappa * d))**2
    res_TM = 4 * k * kappa ** 2 / 2 / pi * rTM_L * rTM_R * exp(-2 * kappa * d) / (1 - rTM_L * rTM_R * exp(-2 * kappa * d))**2
    return res_TE, res_TM

def k0_func_pressuregradient(k0, d, epsm_func, rL, rR, epsrel=1.e-8, epsabs=0.0, mum_func=None):
    """
    Casimir pressure gradient contribution at a given wave number k0.

    Parameters
    ----------
    k0 : float
        vacuum wave number
    d : float
        separation between the two plates
    epsm_func : function
        dielectric function of the medium
    rL, rR : float
        reflection coefficient of the left and right plate, respectively
    mum_func : function or None
        magnetic permeability of the medium. If None, the medium is treated as non-magnetic.

    Returns
    -------
    numpy array of two floats
        result for TE and TM polarization
    """
    return _integrate_k0_contribution(k_integrand_pressuregradient, k0, d, epsm_func, rL, rR, epsrel=epsrel, epsabs=epsabs, mum_func=mum_func)
