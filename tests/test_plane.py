from math import sqrt

import numpy as np
from numpy.testing import assert_allclose

from califorcia.plane import (
    def_fresnel_coefficients,
    def_reflection_coeff,
    evaluate_eps_stack,
    fresnel_coefficients_finite,
    kappa,
    reflection_coefficients_finite,
)
from califorcia.materials import gold_drude, gold_plasma, pec, teflon, vacuum


def test_kappa_zero_frequency_depends_on_materialclass():
    kval = 2.0
    assert_allclose(kappa(teflon, 0.0, kval), kval)
    assert_allclose(kappa(gold_drude, 0.0, kval), kval)
    assert_allclose(kappa(gold_plasma, 0.0, kval), sqrt((gold_plasma.wp / 299792458.0) ** 2 + kval**2))


def test_fresnel_coefficients_for_pec_are_exact():
    fresnel = def_fresnel_coefficients(vacuum, pec)
    assert fresnel(0.0, 1.0) == (1.0, -1.0)
    assert fresnel(1.0, 1.0) == (1.0, -1.0)


def test_dielectric_dielectric_zero_frequency_te_mode_vanishes():
    fresnel = def_fresnel_coefficients(vacuum, teflon)
    r_tm, r_te = fresnel(0.0, 1.0)
    assert_allclose(r_te, 0.0, atol=0.0)
    assert r_tm > 0.0


def test_zero_thickness_multilayer_reflection_reduces_to_single_interface():
    multilayer = def_reflection_coeff(vacuum, [teflon, gold_drude], [0.0])
    single = def_reflection_coeff(vacuum, [gold_drude], [])
    assert_allclose(multilayer(1.2, 0.7), single(1.2, 0.7), rtol=1e-12)


def test_identical_material_layers_reduce_to_same_reflection():
    multilayer = def_reflection_coeff(vacuum, [gold_drude, gold_drude, gold_drude, gold_drude], [5e-9, 7e-9, 9e-9])
    single = def_reflection_coeff(vacuum, [gold_drude], [])
    assert_allclose(multilayer(1.2, 0.7), single(1.2, 0.7), rtol=1e-12)


def test_many_identical_material_layers_reduce_to_same_reflection():
    multilayer = def_reflection_coeff(
        vacuum,
        [gold_drude, gold_drude, gold_drude, gold_drude, gold_drude, gold_drude],
        [5e-9, 7e-9, 9e-9, 11e-9, 13e-9],
    )
    single = def_reflection_coeff(vacuum, [gold_drude], [])
    assert_allclose(multilayer(1.2, 0.7), single(1.2, 0.7), rtol=1e-12)


def test_finite_frequency_fresnel_helper_matches_public_interface():
    k0 = 1.2
    k = 0.7
    public = def_fresnel_coefficients(vacuum, teflon)
    helper = fresnel_coefficients_finite(vacuum.epsilon(k0 * 299792458.0), teflon.epsilon(k0 * 299792458.0), k0, k)
    assert_allclose(public(k0, k), helper, rtol=1e-12)


def test_finite_frequency_reflection_helper_matches_public_interface():
    k0 = 1.2
    k = 0.7
    public = def_reflection_coeff(vacuum, [teflon, gold_drude, teflon], [5e-9, 7e-9])
    eps_layers = evaluate_eps_stack([teflon, gold_drude, teflon], k0 * 299792458.0)
    helper = reflection_coefficients_finite(vacuum.epsilon(k0 * 299792458.0), eps_layers, np.array([5e-9, 7e-9]), k0, k)
    assert_allclose(public(k0, k), helper, rtol=1e-12)
