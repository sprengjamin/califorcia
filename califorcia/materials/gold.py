from scipy.constants import hbar
from scipy.constants import e as eV
from ..models import DrudeModel

wp = 9*eV/hbar # plasma frequency
gamma = 0.035*eV/hbar # damping

epsilon = DrudeModel(wp, gamma)
materialclass = epsilon.materialclass
