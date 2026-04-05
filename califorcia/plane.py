from math import sqrt, exp

import numpy as np
from scipy.constants import c

from ._optional_numba import optional_njit


@optional_njit(cache=True)
def _combine_reflection_coefficients(r_left, r_right, kappa_layer, thickness):
    phase = exp(-2 * kappa_layer * thickness)
    rTM_left, rTE_left = r_left
    rTM_right, rTE_right = r_right
    rTM = (rTM_left + rTM_right * phase) / (1 + rTM_left * rTM_right * phase)
    rTE = (rTE_left + rTE_right * phase) / (1 + rTE_left * rTE_right * phase)
    return rTM, rTE


@optional_njit(cache=True)
def kappa_finite(eps, k0, k):
    return sqrt(eps * k0**2 + k**2)


def kappa_zero(mat, k):
    if mat.materialclass == "dielectric":
        return k
    if mat.materialclass == "drude":
        return k
    if mat.materialclass == "plasma":
        Kp = mat.wp / c
        return sqrt(Kp**2 + k**2)
    raise ValueError(f"Unsupported materialclass '{mat.materialclass}' in zero-frequency kappa.")


def fresnel_coefficients_zero(mat1, mat2, k):
    rTM, rTE = 0.0, 0.0
    if mat1.materialclass == "pec":
        return -1.0, 1.0
    if mat2.materialclass == "pec":
        return 1.0, -1.0
    if mat1.materialclass == "dielectric":
        if mat2.materialclass == "dielectric":
            eps1 = mat1.epsilon(0.0)
            eps2 = mat2.epsilon(0.0)
            rTM = (eps2 - eps1) / (eps2 + eps1)
            rTE = 0.0
        elif mat2.materialclass == "drude":
            rTM = 1.0
            rTE = 0.0
        elif mat2.materialclass == "plasma":
            Kp = mat2.wp / c
            rTM = 1.0
            rTE = (k - sqrt(Kp**2 + k**2)) / (k + sqrt(Kp**2 + k**2))
    elif mat1.materialclass == "drude":
        if mat2.materialclass == "dielectric":
            rTM = -1.0
            rTE = 0.0
        elif mat2.materialclass == "drude":
            wp1 = mat1.wp
            gamma1 = mat1.gamma
            wp2 = mat2.wp
            gamma2 = mat2.gamma
            rTM = (gamma1 * wp2**2 - gamma2 * wp1**2) / (gamma1 * wp2**2 + gamma2 * wp1**2)
            rTE = 0.0
    elif mat1.materialclass == "plasma":
        if mat2.materialclass == "dielectric":
            Kp = mat1.wp / c
            rTM = -1.0
            rTE = -(k - sqrt(Kp**2 + k**2)) / (k + sqrt(Kp**2 + k**2))
        elif mat2.materialclass == "plasma":
            Kp1 = mat1.wp / c
            Kp2 = mat2.wp / c
            q1 = sqrt(Kp1**2 + k**2)
            q2 = sqrt(Kp2**2 + k**2)
            rTM = (Kp2**2 * q1 - Kp1**2 * q2) / (Kp2**2 * q1 + Kp1**2 * q2)
            rTE = (q1 - q2) / (q1 + q2)
    return rTM, rTE


@optional_njit(cache=True)
def fresnel_coefficients_finite(eps1, eps2, k0, k):
    q1 = kappa_finite(eps1, k0, k)
    q2 = kappa_finite(eps2, k0, k)
    rTM = (eps2 * q1 - eps1 * q2) / (eps1 * q2 + eps2 * q1)
    rTE = (q1 - q2) / (q2 + q1)
    return rTM, rTE


@optional_njit(cache=True)
def reflection_coefficients_finite(eps_medium, eps_layers, thicknesses, k0, k):
    nlayers = len(eps_layers)
    if nlayers == 1:
        return fresnel_coefficients_finite(eps_medium, eps_layers[0], k0, k)

    effective_reflection = fresnel_coefficients_finite(eps_layers[nlayers - 2], eps_layers[nlayers - 1], k0, k)
    for idx in range(nlayers - 3, -1, -1):
        interface_reflection = fresnel_coefficients_finite(eps_layers[idx], eps_layers[idx + 1], k0, k)
        effective_reflection = _combine_reflection_coefficients(
            interface_reflection,
            effective_reflection,
            kappa_finite(eps_layers[idx + 1], k0, k),
            thicknesses[idx + 1],
        )

    first_interface = fresnel_coefficients_finite(eps_medium, eps_layers[0], k0, k)
    return _combine_reflection_coefficients(
        first_interface,
        effective_reflection,
        kappa_finite(eps_layers[0], k0, k),
        thicknesses[0],
    )


def evaluate_eps_stack(materials, xi):
    return np.asarray([material.epsilon(xi) for material in materials], dtype=np.float64)


def def_reflection_coeff(medium, materials, thicknesses):
    """
    Define reflection coefficients of the plane by specifying medium and materials of the plane and thickness of the
    coating layers.
    """
    nlayers = len(materials)
    interface_coefficients = [def_fresnel_coefficients(medium, materials[0])]
    interface_coefficients.extend(
        def_fresnel_coefficients(materials[idx], materials[idx + 1])
        for idx in range(nlayers - 1)
    )

    if nlayers == 1:
        return interface_coefficients[0]

    def reflection_coeff(k0, k):
        effective_reflection = interface_coefficients[-1](k0, k)

        for idx in range(nlayers - 2, 0, -1):
            effective_reflection = _combine_reflection_coefficients(
                interface_coefficients[idx](k0, k),
                effective_reflection,
                kappa(materials[idx], k0, k),
                thicknesses[idx],
            )

        return _combine_reflection_coefficients(
            interface_coefficients[0](k0, k),
            effective_reflection,
            kappa(materials[0], k0, k),
            thicknesses[0],
        )

    return reflection_coeff


def kappa(mat, k0, k):
    if k0 == 0.0:
        return kappa_zero(mat, k)
    return kappa_finite(mat.epsilon(k0 * c), k0, k)


def def_fresnel_coefficients(mat1, mat2):
    """
    Defines Fresnel reflection coefficients for a halfspace.
    """

    def fresnel_coefficients(k0, k):
        if mat1.materialclass == "pec":
            return -1.0, 1.0
        if mat2.materialclass == "pec":
            return 1.0, -1.0
        if k0 == 0.0:
            return fresnel_coefficients_zero(mat1, mat2, k)
        return fresnel_coefficients_finite(mat1.epsilon(k0 * c), mat2.epsilon(k0 * c), k0, k)

    return fresnel_coefficients
