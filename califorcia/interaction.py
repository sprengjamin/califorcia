import numpy as np
from scipy.integrate import quad_vec
from math import sqrt, exp, inf, log1p, pi
from scipy.constants import c

from ._optional_numba import optional_njit
from .plane import evaluate_eps_stack, reflection_coefficients_finite


def k_integrand_energy(k, k0, d, epsm, rL, rR):
    kappa = sqrt(epsm * k0 ** 2 + k ** 2)
    rTM_L, rTE_L = rL(k0, k)
    rTM_R, rTE_R = rR(k0, k)
    res_TE = k / 2 / pi * log1p(- rTE_L * rTE_R * exp(-2 * kappa * d))
    res_TM = k / 2 / pi * log1p(- rTM_L * rTM_R * exp(-2 * kappa * d))
    return res_TE, res_TM


def k_integrand_pressure(k, k0, d, epsm, rL, rR):
    kappa = sqrt(epsm * k0 ** 2 + k ** 2)
    rTM_L, rTE_L = rL(k0, k)
    rTM_R, rTE_R = rR(k0, k)
    res_TE = -2 * k * kappa / 2 / pi * rTE_L * rTE_R * exp(-2 * kappa * d) / (1 - rTE_L * rTE_R * exp(-2 * kappa * d))
    res_TM = -2 * k * kappa / 2 / pi * rTM_L * rTM_R * exp(-2 * kappa * d) / (1 - rTM_L * rTM_R * exp(-2 * kappa * d))
    return res_TE, res_TM


def k_integrand_pressuregradient(k, k0, d, epsm, rL, rR):
    kappa = sqrt(epsm * k0 ** 2 + k ** 2)
    rTM_L, rTE_L = rL(k0, k)
    rTM_R, rTE_R = rR(k0, k)
    res_TE = 4 * k * kappa ** 2 / 2 / pi * rTE_L * rTE_R * exp(-2 * kappa * d) / (1 - rTE_L * rTE_R * exp(-2 * kappa * d))**2
    res_TM = 4 * k * kappa ** 2 / 2 / pi * rTM_L * rTM_R * exp(-2 * kappa * d) / (1 - rTM_L * rTM_R * exp(-2 * kappa * d))**2
    return res_TE, res_TM


@optional_njit(cache=True)
def _finite_energy_integrand(k, k0, d, epsm, eps_layers_L, thicknesses_L, eps_layers_R, thicknesses_R):
    kappa = sqrt(epsm * k0**2 + k**2)
    rTM_L, rTE_L = reflection_coefficients_finite(epsm, eps_layers_L, thicknesses_L, k0, k)
    rTM_R, rTE_R = reflection_coefficients_finite(epsm, eps_layers_R, thicknesses_R, k0, k)
    res_TE = k / 2 / pi * log1p(-rTE_L * rTE_R * exp(-2 * kappa * d))
    res_TM = k / 2 / pi * log1p(-rTM_L * rTM_R * exp(-2 * kappa * d))
    return res_TE, res_TM


@optional_njit(cache=True)
def _finite_pressure_integrand(k, k0, d, epsm, eps_layers_L, thicknesses_L, eps_layers_R, thicknesses_R):
    kappa = sqrt(epsm * k0**2 + k**2)
    rTM_L, rTE_L = reflection_coefficients_finite(epsm, eps_layers_L, thicknesses_L, k0, k)
    rTM_R, rTE_R = reflection_coefficients_finite(epsm, eps_layers_R, thicknesses_R, k0, k)
    res_TE = -2 * k * kappa / 2 / pi * rTE_L * rTE_R * exp(-2 * kappa * d) / (1 - rTE_L * rTE_R * exp(-2 * kappa * d))
    res_TM = -2 * k * kappa / 2 / pi * rTM_L * rTM_R * exp(-2 * kappa * d) / (1 - rTM_L * rTM_R * exp(-2 * kappa * d))
    return res_TE, res_TM


@optional_njit(cache=True)
def _finite_pressuregradient_integrand(k, k0, d, epsm, eps_layers_L, thicknesses_L, eps_layers_R, thicknesses_R):
    kappa = sqrt(epsm * k0**2 + k**2)
    rTM_L, rTE_L = reflection_coefficients_finite(epsm, eps_layers_L, thicknesses_L, k0, k)
    rTM_R, rTE_R = reflection_coefficients_finite(epsm, eps_layers_R, thicknesses_R, k0, k)
    res_TE = 4 * k * kappa**2 / 2 / pi * rTE_L * rTE_R * exp(-2 * kappa * d) / (1 - rTE_L * rTE_R * exp(-2 * kappa * d))**2
    res_TM = 4 * k * kappa**2 / 2 / pi * rTM_L * rTM_R * exp(-2 * kappa * d) / (1 - rTM_L * rTM_R * exp(-2 * kappa * d))**2
    return res_TE, res_TM


def _integrate_k0_contribution(integrand, k0, d, epsm_func, rL, rR, epsrel=1.e-8, epsabs=0.0):
    epsm = epsm_func(k0 * c)
    f = lambda t: np.array(integrand(t / d, k0, d, epsm, rL, rR)) / d
    return quad_vec(f, 0, inf, epsrel=epsrel, epsabs=epsabs)[0]


def _integrate_k0_contribution_finite(integrand, k0, d, medium, matL, deltaL, matR, deltaR, epsrel=1.e-8, epsabs=0.0):
    xi = k0 * c
    epsm = medium.epsilon(xi)
    eps_layers_L = evaluate_eps_stack(matL, xi)
    eps_layers_R = evaluate_eps_stack(matR, xi)
    thicknesses_L = np.asarray(deltaL, dtype=np.float64)
    thicknesses_R = np.asarray(deltaR, dtype=np.float64)
    f = lambda t: np.array(integrand(t / d, k0, d, epsm, eps_layers_L, thicknesses_L, eps_layers_R, thicknesses_R)) / d
    return quad_vec(f, 0, inf, epsrel=epsrel, epsabs=epsabs)[0]


def _supports_compiled_finite_path(medium, matL, matR):
    if medium.materialclass == "pec":
        return False
    return all(material.materialclass != "pec" for material in matL) and all(material.materialclass != "pec" for material in matR)


def k0_func_energy(k0, d, medium, matL, deltaL, matR, deltaR, rL_zero, rR_zero, epsrel=1.e-8, epsabs=0.0):
    if k0 == 0.0:
        return _integrate_k0_contribution(k_integrand_energy, k0, d, medium.epsilon, rL_zero, rR_zero, epsrel=epsrel, epsabs=epsabs)
    if not _supports_compiled_finite_path(medium, matL, matR):
        return _integrate_k0_contribution(k_integrand_energy, k0, d, medium.epsilon, rL_zero, rR_zero, epsrel=epsrel, epsabs=epsabs)
    return _integrate_k0_contribution_finite(_finite_energy_integrand, k0, d, medium, matL, deltaL, matR, deltaR, epsrel=epsrel, epsabs=epsabs)


def k0_func_pressure(k0, d, medium, matL, deltaL, matR, deltaR, rL_zero, rR_zero, epsrel=1.e-8, epsabs=0.0):
    if k0 == 0.0:
        return _integrate_k0_contribution(k_integrand_pressure, k0, d, medium.epsilon, rL_zero, rR_zero, epsrel=epsrel, epsabs=epsabs)
    if not _supports_compiled_finite_path(medium, matL, matR):
        return _integrate_k0_contribution(k_integrand_pressure, k0, d, medium.epsilon, rL_zero, rR_zero, epsrel=epsrel, epsabs=epsabs)
    return _integrate_k0_contribution_finite(_finite_pressure_integrand, k0, d, medium, matL, deltaL, matR, deltaR, epsrel=epsrel, epsabs=epsabs)


def k0_func_pressuregradient(k0, d, medium, matL, deltaL, matR, deltaR, rL_zero, rR_zero, epsrel=1.e-8, epsabs=0.0):
    if k0 == 0.0:
        return _integrate_k0_contribution(k_integrand_pressuregradient, k0, d, medium.epsilon, rL_zero, rR_zero, epsrel=epsrel, epsabs=epsabs)
    if not _supports_compiled_finite_path(medium, matL, matR):
        return _integrate_k0_contribution(k_integrand_pressuregradient, k0, d, medium.epsilon, rL_zero, rR_zero, epsrel=epsrel, epsabs=epsabs)
    return _integrate_k0_contribution_finite(_finite_pressuregradient_integrand, k0, d, medium, matL, deltaL, matR, deltaR, epsrel=epsrel, epsabs=epsabs)
