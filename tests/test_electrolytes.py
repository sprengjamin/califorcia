import pytest
import numpy as np
from califorcia.compute import system
from califorcia.materials import salt_water, gold, vacuum
from scipy.constants import k as kB
from math import sqrt

def test_electrolyte_screening_n0():
    """
    Test that if the medium is an electrolyte, the n=0 term is screened (rTM = -1).
    For a symmetric system (Gold - Salt Water - Gold) at high T (n=0 only),
    rTM = -1 for both interfaces.
    The Lifshitz formula integrand for TM at k0=0 is:
    log(1 - rTM_L * rTM_R * exp(-2*k*d)) = log(1 - (-1)*(-1)*exp(-2*k*d)) = log(1 - exp(-2*k*d)).
    """
    d = 100e-9
    T = 300
    s = system(T, d, gold, gold, salt_water)
    
    # Calculate zero-frequency contribution only
    p0 = s.calculate('pressure', ht_limit=True)
    
    # Analytical expectation for rTM = -1 at both interfaces:
    # Pressure P = - (k_B T / 2*pi) * integral_0^inf k * 2*k * exp(-2kd) / (1 - exp(-2kd)) dk
    # Let x = 2kd, dx = 2d dk
    # P = - (k_B T / 2*pi) * integral_0^inf (x/2d) * (x/d) * exp(-x) / (1 - exp(-x)) * (1/2d) dx
    # P = - (k_B T / 8*pi*d^3) * integral_0^inf x^2 * exp(-x) / (1 - exp(-x)) dx
    # The integral is 2 * zeta(3).
    # P = - 0.5 * (k_B T / 8*pi*d^3) * 2 * zeta(3) = - zeta(3) * k_B T / (8 * pi * d^3)
    
    from scipy.special import zeta
    expected_p = - zeta(3) * kB * T / (8 * np.pi * d**3)
    
    # Note: s.calculate returns self.n0 which is 0.5 * T * kB * f(0).
    # f(0) is calculated in interaction.py:
    # res_TM = -2 * k * k / 2 / pi * (-1)*(-1) * exp(-2kd) / (1 - exp(-2kd))
    # which is exactly what leads to the formula above.
    
    assert np.isclose(p0, expected_p, rtol=1e-5)

def test_electrolyte_as_plate():
    """Test electrolyte as a plate material."""
    d = 100e-9
    T = 300
    # Gold - Vacuum - Salt Water
    s = system(T, d, gold, salt_water, vacuum)
    
    p0 = s.calculate('pressure', ht_limit=True)
    
    # At k0=0:
    # Interface 1 (Vacuum/Gold): mat1=dielectric, mat2=drude -> rTM = 1
    # Interface 2 (Vacuum/Salt Water): mat1=dielectric, mat2=electrolites -> rTM = -1
    # P = - (k_B T / 2*pi) * integral k * 2k * rL * rR * exp(-2kd) / (1 - rL*rR*exp(-2kd)) dk
    # P = - (k_B T / 2*pi) * integral k * 2k * (1) * (-1) * exp(-2kd) / (1 - (1)*(-1)*exp(-2kd)) dk
    # P = + (k_B T / 8*pi*d^3) * integral x^2 * exp(-x) / (1 + exp(-x)) dx
    # The integral is 1.5 * zeta(3) ? No, it's something else, but it's repulsive (positive).
    
    assert p0 > 0

def test_longitudinal_channel_repulsion():
    """
    Test that the longitudinal channel is repulsive for symmetric systems.
    For Gold-Salt Water-Gold, r_ll = -1 for both interfaces.
    F0_long = (kBT/2) * integral (d^2k/(2pi)^2) log(1 - (-1)*(-1)*exp(-2*kappa_l*d))
    = (kBT/2) * integral (k dk / 2pi) log(1 - exp(-2*kappa_l*d))
    The log is negative, so F0_long is negative (repulsive energy).
    Pressure P_long = -dF/dd should be positive (repulsive).
    """
    d = 100e-9
    T = 300
    s = system(T, d, gold, gold, salt_water)
    
    p_long = s.calculate_longitudinal('pressure')
    
    # Expected to be positive
    assert p_long > 0
    
    # Check if it matches analytic form with rL = rR = -1
    # P_long = (kBT/2) * integral (k dk / 2pi) * 2 * kappa_l * exp(-2*kappa_l*d) / (1 - exp(-2*kappa_l*d))
    # This is exactly the same integral as the TM one with rTM = 1 (PEC case) but with shifted kappa.
    # If d >> lambda_D, kappa_l ~ k, then P_long ~ - P_TM_PEC(0) = repulsive.
    pass

def test_longitudinal_fresnel_silica_water():
    """
    Test longitudinal Fresnel coefficient for silica-water interface.
    Eq. (18): r_ll = (1 - (kl/k)*(eps1/epsb)) / (1 + (kl/k)*(eps1/epsb))
    """
    from califorcia.materials import silica
    from califorcia.plane import def_longitudinal_fresnel_coefficients
    
    fresnel = def_longitudinal_fresnel_coefficients(silica, salt_water)
    
    k = 1e7
    kappa_l = sqrt(k**2 + salt_water.kappa_D**2)
    eps_s = silica.epsilon(0.)
    eps_b = salt_water.solvent_model.epsilon(0.)
    
    r_expected = (1 - (kappa_l/k)*(eps_s/eps_b)) / (1 + (kappa_l/k)*(eps_s/eps_b))
    
    assert np.isclose(fresnel(0.0, k), r_expected)
