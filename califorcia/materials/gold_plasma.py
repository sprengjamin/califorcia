from scipy.constants import hbar
from scipy.constants import e as eV
from ..models import PlasmaModel

wp = 9*eV/hbar # plasma frequency

epsilon = PlasmaModel(wp)
materialclass = epsilon.materialclass
