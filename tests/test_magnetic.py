from math import sqrt

import pytest
from numpy.testing import assert_allclose

from califorcia.compute import system
from califorcia.materials import gold_drude, teflon, vacuum
from califorcia.plane import (
    def_fresnel_coefficients,
    def_reflection_coeff,
    kappa,
    permeability,
)


class NonMagnetic:
    """Frequency-independent dielectric that does not define a permeability."""

    materialclass = "dielectric"

    def __init__(self, eps):
        self._eps = eps

    def epsilon(self, xi):
        return self._eps


class Magnetic:
    """Frequency-independent dielectric and magnetic response."""

    materialclass = "dielectric"

    def __init__(self, eps, mu):
        self._eps = eps
        self._mu = mu

    def epsilon(self, xi):
        return self._eps

    def mu(self, xi):
        return self._mu


class MagneticDrude:
    """Drude material carrying a static permeability."""

    materialclass = "drude"

    def __init__(self, wp, gamma, mu):
        self.wp = wp
        self.gamma = gamma
        self._mu = mu

    def epsilon(self, xi):
        return 1. + self.wp**2/(xi*(xi + self.gamma))

    def mu(self, xi):
        return self._mu


class MagneticPlasma:
    """Plasma material carrying a static permeability."""

    materialclass = "plasma"

    def __init__(self, wp, mu):
        self.wp = wp
        self._mu = mu

    def epsilon(self, xi):
        return 1. + self.wp**2/xi**2

    def mu(self, xi):
        return self._mu


def test_permeability_defaults_to_unity_when_material_defines_none():
    assert permeability(teflon, 1.e15) == 1.
    assert permeability(vacuum, 0.) == 1.
    assert permeability(Magnetic(2., 3.), 1.e15) == 3.


def test_material_without_mu_matches_explicit_unit_permeability():
    implicit = def_fresnel_coefficients(vacuum, NonMagnetic(4.))
    explicit = def_fresnel_coefficients(vacuum, Magnetic(4., 1.))
    for k0, k in [(0., 1.e7), (1.2, 0.7), (3.e6, 2.e7)]:
        assert_allclose(implicit(k0, k), explicit(k0, k), rtol=1e-15)


def test_kappa_includes_permeability():
    eps, mu, k0, k = 4., 3., 1.5, 0.8
    assert_allclose(kappa(Magnetic(eps, mu), k0, k), sqrt(eps*mu*k0**2 + k**2))


def test_kappa_of_magnetic_plasma_at_zero_frequency():
    wp, mu, k = 1.4e16, 5., 2.
    Kp = wp/299792458.
    assert_allclose(kappa(MagneticPlasma(wp, mu), 0., k), sqrt(mu*Kp**2 + k**2))


def test_zero_frequency_te_mode_driven_by_static_permeability():
    eps, mu = 4., 3.
    fresnel = def_fresnel_coefficients(vacuum, Magnetic(eps, mu))
    r_tm, r_te = fresnel(0., 1.e7)
    assert_allclose(r_tm, (eps - 1.)/(eps + 1.))
    assert_allclose(r_te, (mu - 1.)/(mu + 1.))


def test_zero_frequency_limits_stay_finite_for_every_materialclass():
    # mu(0) is finite for real materials, unlike epsilon, which diverges as 1/xi
    # for drude and as 1/xi**2 for plasma. The zero-frequency limit of kappa is
    # therefore finite in every case, so TE needs no case distinction.
    mu, k = 4., 1.e7
    contrast = (mu - 1.)/(mu + 1.)

    for material in (Magnetic(4., mu), MagneticDrude(1.4e16, 5.3e13, mu)):
        assert_allclose(kappa(material, 0., k), k)
        _, r_te = def_fresnel_coefficients(vacuum, material)(0., k)
        assert_allclose(r_te, contrast)

    plasma = MagneticPlasma(1.4e16, mu)
    Kp = plasma.wp/299792458.
    q = sqrt(mu*Kp**2 + k**2)
    assert_allclose(kappa(plasma, 0., k), q)
    _, r_te = def_fresnel_coefficients(vacuum, plasma)(0., k)
    assert_allclose(r_te, (mu*k - q)/(mu*k + q))
    assert abs(r_te) < 1.


def test_magnetic_drude_retains_zero_frequency_te_contribution():
    # A non-magnetic drude material contributes nothing to TE at zero frequency,
    # whereas a magnetic one is governed by the static permeability contrast.
    _, r_te_plain = def_fresnel_coefficients(vacuum, gold_drude)(0., 1.e7)
    assert r_te_plain == 0.

    magnetic = MagneticDrude(gold_drude.wp, gold_drude.gamma, 4.)
    _, r_te_magnetic = def_fresnel_coefficients(vacuum, magnetic)(0., 1.e7)
    assert_allclose(r_te_magnetic, 3./5.)


def test_zero_frequency_te_mode_still_vanishes_without_magnetism():
    fresnel = def_fresnel_coefficients(vacuum, NonMagnetic(4.))
    _, r_te = fresnel(0., 1.e7)
    assert r_te == 0.


def test_electric_magnetic_duality_swaps_polarizations():
    direct = def_fresnel_coefficients(Magnetic(2., 3.), Magnetic(5., 7.))
    dual = def_fresnel_coefficients(Magnetic(3., 2.), Magnetic(7., 5.))
    for k0, k in [(0., 1.e7), (1.2, 0.7), (3.e6, 2.e7)]:
        r_tm, r_te = direct(k0, k)
        dual_tm, dual_te = dual(k0, k)
        assert_allclose(r_tm, dual_te, rtol=1e-14)
        assert_allclose(r_te, dual_tm, rtol=1e-14)


def test_impedance_matched_material_does_not_reflect_at_normal_incidence():
    # A material with eps == mu has the same wave impedance as vacuum, so both
    # reflection coefficients vanish at normal incidence.
    for n in (2., 10., 1.e3):
        fresnel = def_fresnel_coefficients(vacuum, Magnetic(n, n))
        assert_allclose(fresnel(1.5, 0.), (0., 0.), atol=1e-15)


def test_magnetic_layer_of_zero_thickness_reduces_to_single_interface():
    multilayer = def_reflection_coeff(vacuum, [Magnetic(4., 3.), NonMagnetic(9.)], [0.])
    single = def_reflection_coeff(vacuum, [NonMagnetic(9.)], [])
    assert_allclose(multilayer(1.2, 0.7), single(1.2, 0.7), rtol=1e-12)


def test_identical_magnetic_layers_reduce_to_single_interface():
    layer = Magnetic(4., 3.)
    multilayer = def_reflection_coeff(vacuum, [layer, layer, layer], [5e-9, 7e-9])
    single = def_reflection_coeff(vacuum, [layer], [])
    assert_allclose(multilayer(1.2, 0.7), single(1.2, 0.7), rtol=1e-12)


def test_permeability_changes_the_layer_phase():
    magnetic_stack = def_reflection_coeff(vacuum, [Magnetic(4., 3.), NonMagnetic(9.)], [5e-9])
    plain_stack = def_reflection_coeff(vacuum, [NonMagnetic(4.), NonMagnetic(9.)], [5e-9])
    assert magnetic_stack(3.e6, 2.e7) != plain_stack(3.e6, 2.e7)


def test_non_magnetic_system_is_unchanged_by_explicit_unit_permeability():
    implicit = system(T=300., d=1.e-7, matL=NonMagnetic(4.), matR=NonMagnetic(4.), matm=vacuum)
    explicit = system(T=300., d=1.e-7, matL=Magnetic(4., 1.), matR=Magnetic(4., 1.), matm=vacuum)
    assert_allclose(implicit.energy(epsrel=1e-6), explicit.energy(epsrel=1e-6), rtol=1e-12)


def test_magnetic_medium_is_accounted_for():
    reference = system(T=300., d=1.e-7, matL=NonMagnetic(4.), matR=NonMagnetic(4.), matm=vacuum)
    unit_mu = system(T=300., d=1.e-7, matL=NonMagnetic(4.), matR=NonMagnetic(4.), matm=Magnetic(1., 1.))
    magnetic = system(T=300., d=1.e-7, matL=NonMagnetic(4.), matR=NonMagnetic(4.), matm=Magnetic(1., 2.))

    assert_allclose(reference.energy(epsrel=1e-6), unit_mu.energy(epsrel=1e-6), rtol=1e-12)
    assert magnetic.energy(epsrel=1e-6) != pytest.approx(reference.energy(epsrel=1e-6), rel=1e-6)


def test_electric_and_magnetic_plate_repel():
    # Boyer's result: a strongly electric and a strongly magnetic mirror repel,
    # whereas two electric mirrors attract.
    electric = Magnetic(1.e4, 1.)
    magnetic = Magnetic(1., 1.e4)

    mixed = system(T=300., d=1.e-7, matL=electric, matR=magnetic, matm=vacuum)
    alike = system(T=300., d=1.e-7, matL=electric, matR=electric, matm=vacuum)

    assert mixed.energy(epsrel=1e-6) > 0.
    assert alike.energy(epsrel=1e-6) < 0.


def test_non_callable_permeability_is_rejected():
    class BadMu:
        materialclass = "dielectric"
        mu = 2.

        def epsilon(self, xi):
            return 2.

    with pytest.raises(ValueError, match="must be callable"):
        system(T=300., d=1.e-7, matL=BadMu(), matR=vacuum, matm=vacuum)
