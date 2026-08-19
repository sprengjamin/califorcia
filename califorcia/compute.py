import numpy as np
from .plane import def_reflection_coeff, permeability
from .interaction import k0_func_energy, k0_func_pressure, k0_func_pressuregradient
from .frequency_summation import psd_sum, msd_sum
from scipy.constants import k as kB
from scipy.constants import pi, c, hbar
from scipy.integrate import quad
from math import inf

SUPPORTED_MATERIALCLASSES = {"dielectric", "drude", "plasma", "pec"}

class system:
    '''Class that defines the system of two parallel plates.
    '''
    def __init__(self, T, d, matL, matR, matm, deltaL=[], deltaR=[]):
        '''
        System parameters are defined and initialized.

        Parameters
        ----------
        T : float
            Temperature in K
        d : float
            Separation in m
        matL, matR: object or list objects
            Material of left and right plate, respectively. If specified as a list, the first material corresponds to
            the coating facing the medium and so on.
        matm : object
            Material of medium

            Materials may optionally define a magnetic permeability `mu(xi)` on the imaginary frequency axis.
            Materials without it are treated as non-magnetic, i.e. mu = 1.
        deltaL, deltaR : list
            Thicknesses of the coating layers with the first one corresponding to the thickness of the coating facing
            the medium and so on.
        '''
        self.T = T
        self.d = d
        if not isinstance(matL, list):
            self.matL = [matL]
        else:
            self.matL = matL
        self.deltaL = deltaL
        if not isinstance(matR, list):
            self.matR = [matR]
        else:
            self.matR = matR
        self.deltaR = deltaR
        self.matm = matm
        self._validate_inputs()

    def _validate_material(self, material, location):
        if not hasattr(material, "materialclass"):
            raise ValueError(f"Material '{location}' must define 'materialclass'.")

        materialclass = material.materialclass
        if materialclass not in SUPPORTED_MATERIALCLASSES:
            raise ValueError(
                f"Unsupported materialclass '{materialclass}' for material '{location}'. "
                f"Supported values are {sorted(SUPPORTED_MATERIALCLASSES)}."
            )

        if not hasattr(material, "epsilon"):
            raise ValueError(f"Material '{location}' must define 'epsilon(xi)'.")

        if materialclass == "plasma" and not hasattr(material, "wp"):
            raise ValueError(f"Material '{location}' with materialclass 'plasma' must define 'wp'.")

        if hasattr(material, "mu") and not callable(material.mu):
            raise ValueError(
                f"Material '{location}' defines 'mu', which must be callable as 'mu(xi)'. "
                "Materials without 'mu' are treated as non-magnetic."
            )

    def _validate_inputs(self):
        if self.d <= 0.0:
            raise ValueError("The plate separation d must be positive.")

        if not len(self.matL) == len(self.deltaL) + 1:
            raise ValueError("A thickness needs to be assigned to each coating layer on plate L, i.e. len(matL)=len(deltaL)+1 must hold.")
        if not len(self.matR) == len(self.deltaR) + 1:
            raise ValueError("A thickness needs to be assigned to each coating layer on plate R, i.e. len(matR)=len(deltaR)+1 must hold.")
        if any(thickness < 0.0 for thickness in self.deltaL):
            raise ValueError("Coating thicknesses on plate L must be non-negative.")
        if any(thickness < 0.0 for thickness in self.deltaR):
            raise ValueError("Coating thicknesses on plate R must be non-negative.")

        self._validate_material(self.matm, "matm")
        for idx, material in enumerate(self.matL):
            self._validate_material(material, f"matL[{idx}]")
        for idx, material in enumerate(self.matR):
            self._validate_material(material, f"matR[{idx}]")

    def frequency_function(self, observable, epsrel=1.e-8, epsabs=0.0):
        '''
        Defines the frequency summand or integrand within Lifshitz formula based on the specified observable.

        Parameters
        ----------
        observable : str
            either 'energy', 'pressure' or 'pressuregradient'

        Returns
        -------
        function

        '''
        if observable == 'energy':
            func = k0_func_energy
        elif observable == 'pressure':
            func = k0_func_pressure
        elif observable == 'pressuregradient':
            func = k0_func_pressuregradient
        else:
            raise ValueError('Supported values for \'observable\' are either \'energy\', \'pressure\' or \'pressuregradient\'!')

        # define reflection coefficients
        rL = def_reflection_coeff(self.matm, self.matL, self.deltaL)
        rR = def_reflection_coeff(self.matm, self.matR, self.deltaR)

        # define magnetic permeability of the medium
        mum = lambda xi: permeability(self.matm, xi)

        # define frequency (wave vector) integrand/summand
        return lambda k0: func(k0, self.d, self.matm.epsilon, rL, rR, epsrel=epsrel, epsabs=epsabs, mum_func=mum)

    def calculate(self, observable, ht_limit=False, fs='psd', epsrel=1.e-8, epsabs=0.0, N=None):
        '''
        Calculate the Casimir interaction according to the specified observable.

        Parameters
        ----------
        observable : str
            either 'energy', 'pressure' or 'pressuregradient'
        ht_limit : bool
            if set True, the high-temperature limit corresponding to the zero-frequency contribution only is calculated
        fs : str
            Method to be used to calculate the Matsubara frequency summation. Can be set to 'msd' (conventional summation)
            or 'psd' (Pade spectrum decomposition). Default: 'psd'
        epsrel : float
            Target precision for frequency summation
        epsabs : float
            Absolute target precision for the radial and zero-temperature quadratures
        N : int
            Number of terms in the frequency summation. By default, `N=None` and the number is determined automatically
            based on the value of `epsrel`.

        Returns
        -------
        float
            the value of the Casimir interaction
        '''
        self.f = self.frequency_function(observable, epsrel=epsrel, epsabs=epsabs)

        if self.T == 0.:
            # frequency integration
            t_func = lambda t: np.sum(self.f(t / self.d)) / self.d
            return hbar * c / 2 / pi * quad(t_func, 0, inf, epsrel=epsrel, epsabs=epsabs)[0]
        else:
            # frequency summation
            if fs == 'psd':
                fsum = psd_sum
            elif fs == 'msd':
                fsum = msd_sum
            else:
                raise ValueError('Supported values for fs are either \'psd\' or \'msd\'!')

            [self.n0_TE, self.n0_TM] = 0.5 * self.T * kB * self.f(0.)
            self.n0 = self.n0_TE + self.n0_TM
            if ht_limit: return self.n0
            [self.n1_TE, self.n1_TM] = fsum(self.T, self.d, self.f, epsrel=epsrel, order=N)
            self.n1 = self.n1_TE + self.n1_TM
            return self.n0 + self.n1

    def energy(self, ht_limit=False, fs='psd', epsrel=1.e-8, epsabs=0.0, N=None):
        # Calculate the Casimir energy per area
        return self.calculate('energy', ht_limit=ht_limit, fs=fs, epsrel=epsrel, epsabs=epsabs, N=N)

    def pressure(self, ht_limit=False, fs='psd', epsrel=1.e-8, epsabs=0.0, N=None):
        # Calculate the Casimir pressure
        return self.calculate('pressure', ht_limit=ht_limit, fs=fs, epsrel=epsrel, epsabs=epsabs, N=N)

    def pressuregradient(self, ht_limit=False, fs='psd', epsrel=1.e-8, epsabs=0.0, N=None):
        # Calculate the Casimir pressure
        return self.calculate('pressuregradient', ht_limit=ht_limit, fs=fs, epsrel=epsrel, epsabs=epsabs, N=N)
