from scipy.constants import hbar
from scipy.constants import e as eV
from ..models import DrudeModel, LorentzModel, CombinedModel

# Silicon part
eps0 = 11.87
epsinf = 1.035
w0 = 6.6e15
f_si = eps0 - epsinf
si_model = LorentzModel(f_si, w0, 0.0, eps_inf=epsinf)

# Doping part
wp = 3.6151e14 # plasma frequency
gamma = 7.868e13 # damping
drude_model = DrudeModel(wp, gamma)

epsilon = CombinedModel([si_model, drude_model], materialclass="drude")
materialclass = epsilon.materialclass
