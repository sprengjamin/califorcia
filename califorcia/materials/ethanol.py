import numpy as np
from scipy.constants import e as eV
from scipy.constants import hbar
from ..models import DielectricModel

data = np.array([[9.57e-1, 1.62e0, 1.40e-1, 1.26e-1, 4.16e-1, 2.44e-1, 7.10e-2],
                 [1.62e-3, 4.32e-3, 1.12e-1, 6.87e0, 1.52e1, 1.56e1, 4.38e1]])

f_list = data[0]
w_list = data[1]*eV/hbar
eps_static = 24.3

def ethanol_eps(xi, f_list, w_list, eps_static):
    if xi == 0.:
        return eps_static
    else:
        return 1.0 + np.sum(f_list / (1.0 + (xi/w_list)**2))

epsilon = DielectricModel(ethanol_eps, f_list=f_list, w_list=w_list, eps_static=eps_static)
materialclass = epsilon.materialclass
