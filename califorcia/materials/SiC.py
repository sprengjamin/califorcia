from scipy.constants import c, pi
from ..models import LorentzModel

cminv_to_radpersec = 2*pi*100*c
wL = 969*cminv_to_radpersec
wT = 793*cminv_to_radpersec
gamma = 4.76*cminv_to_radpersec
eps_inf = 6.7

# (wL^2 - wT^2)/(wT^2 + xi^2 + gamma*xi)
# f = (wL^2 - wT^2)/wT^2 * eps_inf
f = (wL**2 - wT**2) / wT**2 * eps_inf

epsilon = LorentzModel(f, wT, gamma, eps_inf=eps_inf)
materialclass = epsilon.materialclass
