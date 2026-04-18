import numpy as np
from scipy.constants import e as eV
from scipy.constants import hbar
from ..models import LorentzModel

data = np.array([[9.30e-3, 1.83e-2, 1.39e-1, 1.12e-1, 1.95e-1, 4.38e-1, 1.06e-1, 3.86e-2],
                 [3.00e-4, 7.60e-3, 5.57e-2, 1.26e-1, 6.71e0, 1.86e1, 4.21e1, 7.76e1]])

f_list = data[0]
w_list = data[1]*eV/hbar
g_list = np.zeros_like(f_list)

epsilon = LorentzModel(f_list, w_list, g_list)
materialclass = epsilon.materialclass
