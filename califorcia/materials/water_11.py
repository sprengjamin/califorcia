import numpy as np
from scipy.constants import hbar, e as eV
from ..models import LorentzModel, DebyeModel, CombinedModel

# 11-oscillator model for water (Ref: EPL 112, 44001 (2015) and Segelstein)
# Values for C_j and omega_j (eV)
C_j = np.array([0.771, 0.0524, 0.384, 0.040, 0.010, 0.170, 0.088, 0.138, 0.054, 0.016, 0.005])
w_j_eV = np.array([0.021, 0.068, 0.10, 0.20, 0.40, 8.22, 10.0, 11.4, 13.0, 14.9, 18.5])

# Frequency conversion: eV to rad/s
w_j = w_j_eV * eV / hbar

# Lorentz models (using zero damping as per common oscillator summation form)
lorentz = LorentzModel(C_j, w_j, np.zeros_like(w_j))

# Debye microwave relaxation
eps_s = 78.3
eps_mw = 1.0 + np.sum(C_j) # Value after all IR/UV contributions
tau = 9.4e-12 # [s] Relaxation time

# DebyeModel(D, I, tau) returns I + (D - I) / (1 + xi*tau)
# We want: 1.0 + sum_lorentz + (eps_s - eps_mw) / (1 + xi*tau)
# CombinedModel([lorentz, debye]) returns lorentz.eps + debye.eps - 1.0
# = (1 + sum) + (I + (D-I)/(1+xi*tau)) - 1.0
# If we set I = 1.0 and D = 1.0 + (eps_s - eps_mw), we get 1 + sum + (eps_s - eps_mw)/(1+xi*tau)
debye = DebyeModel(1.0 + (eps_s - eps_mw), 1.0, tau)

epsilon = CombinedModel([lorentz, debye])
materialclass = epsilon.materialclass
