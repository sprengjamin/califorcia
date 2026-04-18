from ..models import LorentzModel, DebyeModel, CombinedModel

# Parameters for water solvent
# Lorentz part
f1, w1, g1 = 0.77, 2.79545e+16, 2.05101e+16
lorentz = LorentzModel(f1, w1, g1)

# Debye part
epsD, epsInf, tau = 78.3, 1.8430, 7.76690e-12
# We want contribution (epsD - epsInf) / (1 + xi*tau)
# DebyeModel(D, I, tau) returns I + (D - I) / (1 + xi*tau)
# If we set I = 1.0 and D = (epsD - epsInf) + 1.0, we get 1.0 + (epsD - epsInf) / (1 + xi*tau)
debye = DebyeModel(epsD - epsInf + 1.0, 1.0, tau)

# Combined solvent model (Water)
# CombinedModel([m1, m2]) returns m1.eps + m2.eps - 1.0
# (1.0 + L) + (1.0 + D_term - 1.0) = 1.0 + L + D_term
epsilon = CombinedModel([lorentz, debye])
materialclass = epsilon.materialclass
