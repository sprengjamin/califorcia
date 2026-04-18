import numpy as np
from scipy.integrate import quad_vec
from math import sqrt, exp, inf, log1p, pi
from scipy.constants import c

def k_integrand_longitudinal_energy(k, k0, d, matm, rL, rR):
    """
    Integrand for the longitudinal scattering channel Casimir energy at k0=0.
    See Eq. (17) in Phys. Rev. A 111, 012816 (2025).
    """
    # kappa_l = sqrt(k^2 + 1/lambda_D^2)
    # 1/lambda_D is matm.kappa_D
    kappa_l = sqrt(k**2 + matm.kappa_D**2)
    r_ll_L = rL(0.0, k)
    r_ll_R = rR(0.0, k)
    
    # The term in the log is (1 - r_ll_L * r_ll_R * exp(-2 * kappa_l * d))
    return k / 2 / pi * log1p(- r_ll_L * r_ll_R * exp(-2 * kappa_l * d))

def _integrate_k0_longitudinal_contribution(integrand, d, matm, rL, rR, epsrel=1.e-8, epsabs=0.0):
    # For longitudinal, only k0=0 is relevant.
    f = lambda t: integrand(t / d, 0.0, d, matm, rL, rR) / d
    return quad_vec(f, 0, inf, epsrel=epsrel, epsabs=epsabs)[0]


def k0_func_longitudinal_energy(k0, d, matm, rL, rR, epsrel=1.e-8, epsabs=0.0):
    """
    Casimir free energy contribution at k0=0 for longitudinal channel.
    """
    if k0 != 0.0:
        return 0.0
    return _integrate_k0_longitudinal_contribution(k_integrand_longitudinal_energy, d, matm, rL, rR, epsrel=epsrel, epsabs=epsabs)

def k_integrand_longitudinal_pressure(k, k0, d, matm, rL, rR):
    """
    Integrand for the longitudinal scattering channel Casimir pressure at k0=0.
    """
    kappa_l = sqrt(k**2 + matm.kappa_D**2)
    r_ll_L = rL(0.0, k)
    r_ll_R = rR(0.0, k)
    
    # Factor 2*kappa_l from derivative of exponent? 
    # Standard Lifshitz pressure: sum_alpha integral k * kappa / pi * ...
    # Eq (17) for energy is F = (kBT/2) * integral (d^2k/(2pi)^2) log(...)
    # Pressure P = -dF/dD = - (kBT/2) * integral (d^2k/(2pi)^2) [ (-2*kappa_l) * rL*rR*exp(-2*kappa_l*d) / (1 - rL*rR*exp(-2*kappa_l*d)) ]
    # P = (kBT/2) * integral (k dk / 2pi) * 2 * kappa_l * rL*rR*exp(-2*kappa_l*d) / (1 - rL*rR*exp(-2*kappa_l*d))
    # P = integral (k * kappa_l / 2pi) * ...
    
    num = r_ll_L * r_ll_R * exp(-2 * kappa_l * d)
    res = 2 * k * kappa_l / 2 / pi * num / (1 - num)
    return res

def k0_func_longitudinal_pressure(k0, d, matm, rL, rR, epsrel=1.e-8, epsabs=0.0):
    """
    Casimir pressure contribution at k0=0 for longitudinal channel.
    """
    if k0 != 0.0:
        return 0.0
    return _integrate_k0_longitudinal_contribution(k_integrand_longitudinal_pressure, d, matm, rL, rR, epsrel=epsrel, epsabs=epsabs)
