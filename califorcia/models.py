import numpy as np
from scipy.constants import hbar, c
from scipy.constants import e as eV
from scipy.constants import k as kB
from math import inf

class MaterialModel:
    """Base class for all material models."""
    def __init__(self, materialclass):
        self.materialclass = materialclass

    def epsilon(self, xi):
        raise NotImplementedError("Subclasses must implement epsilon(xi)")

    def __call__(self, xi):
        return self.epsilon(xi)

class VacuumModel(MaterialModel):
    def __init__(self):
        super().__init__("dielectric")

    def epsilon(self, xi):
        return 1.0

class PECModel(MaterialModel):
    def __init__(self):
        super().__init__("pec")

    def epsilon(self, xi):
        return inf

class DrudeModel(MaterialModel):
    def __init__(self, wp, gamma):
        super().__init__("drude")
        self.wp = wp
        self.gamma = gamma

    def epsilon(self, xi):
        if xi == 0:
            return inf
        return 1.0 + self.wp**2 / (xi * (xi + self.gamma))

class PlasmaModel(MaterialModel):
    def __init__(self, wp):
        super().__init__("plasma")
        self.wp = wp

    def epsilon(self, xi):
        if xi == 0:
            return inf
        return 1.0 + self.wp**2 / xi**2

class DielectricModel(MaterialModel):
    """Generic dielectric model for various representations (Lorentz, Debye, etc.)."""
    def __init__(self, func, **kwargs):
        super().__init__("dielectric")
        self.func = func
        self.params = kwargs

    def epsilon(self, xi):
        return self.func(xi, **self.params)

class ConstantModel(MaterialModel):
    def __init__(self, eps, materialclass="dielectric"):
        super().__init__(materialclass)
        self.eps = eps

    def epsilon(self, xi):
        return self.eps

class LorentzModel(MaterialModel):
    def __init__(self, f_list, w_list, g_list, eps_inf=1.0):
        super().__init__("dielectric")
        self.f_list = np.atleast_1d(f_list)
        self.w_list = np.atleast_1d(w_list)
        self.g_list = np.atleast_1d(g_list)
        self.eps_inf = eps_inf

    def epsilon(self, xi):
        # Implementation of Matsubara-frequency (imaginary) Lorentz model
        return self.eps_inf + np.sum(self.f_list * self.w_list**2 / (self.w_list**2 + xi**2 + self.g_list * xi))

class SellmeierModel(MaterialModel):
    def __init__(self, B_list, C_list):
        """Sellmeier coefficients: B_i and resonance wavelengths squared C_i (in m^2)."""
        super().__init__("dielectric")
        self.B_list = np.atleast_1d(B_list)
        self.C_list = np.atleast_1d(C_list)

    def epsilon(self, xi):
        if xi == 0:
            return 1.0 + np.sum(self.B_list)
        import scipy.constants as const
        l_sq = (2 * np.pi * const.c / xi)**2
        return 1.0 + np.sum(self.B_list * l_sq / (l_sq + self.C_list))

class DebyeModel(MaterialModel):
    def __init__(self, epsD, epsInf, tau):
        super().__init__("dielectric")
        self.epsD = epsD
        self.epsInf = epsInf
        self.tau = tau

    def epsilon(self, xi):
        return self.epsInf + (self.epsD - self.epsInf) / (1.0 + xi * self.tau)

class CombinedModel(MaterialModel):
    def __init__(self, models, materialclass="dielectric"):
        super().__init__(materialclass)
        self.models = models

    def epsilon(self, xi):
        # We assume the first model provides the base epsilon_infinity if multiple models are combined.
        # Or we just sum the contributions relative to 1.0.
        res = self.models[0].epsilon(xi)
        for m in self.models[1:]:
            res += (m.epsilon(xi) - 1.0)
        return res

class ElectrolyteModel(MaterialModel):
    def __init__(self, solvent_model, kappa_D, gamma=1e11, T=300.0, ion_mass=6.492e-26):
        """
        Electrolyte model including solvent dielectric response and ionic screening.
        
        solvent_model: An instance of MaterialModel (usually DielectricModel)
        kappa_D: Debye screening length [1/m]
        gamma: Effective damping for Drude-like behavior at n=0.
        T: Temperature [K]
        ion_mass: Mass of the lightest ion [kg] (default is K+ approx 39 amu)
        """
        super().__init__("electrolites")
        self.solvent_model = solvent_model
        self.kappa_D = kappa_D
        self.gamma = gamma
        self.T = T
        self.ion_mass = ion_mass

    def epsilon(self, xi):
        return self.solvent_model.epsilon(xi)

    @property
    def wp(self):
        """Physical plasma frequency for electrolytes based on thermal velocity."""
        v_th = np.sqrt(kB * self.T / self.ion_mass)
        eps_b0 = self.solvent_model.epsilon(0.0)
        return np.sqrt(eps_b0) * v_th * self.kappa_D
