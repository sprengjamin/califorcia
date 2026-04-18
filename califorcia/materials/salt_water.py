from ..models import ElectrolyteModel
from .water import epsilon as water_solvent

# Electrolyte model (Salt Water)
# kappa_D: Debye screening length [1/m]. 
kappa_D = 1.0e8 # [1/m] Placeholder value, updated in simulations
# Effective ionic damping gamma = 1e11 s^-1 as suggested for Ref 2
gamma_ion = 1e11 

# Lightweight ion mass (approx K+ mass ~ 6.49e-26 kg)
m_ion = 6.492e-26

epsilon = ElectrolyteModel(water_solvent, kappa_D, gamma=gamma_ion, ion_mass=m_ion)
materialclass = epsilon.materialclass
solvent_model = epsilon.solvent_model

# Expose properties for the Fresnel logic
@property
def wp():
    return epsilon.wp

@property
def gamma():
    return epsilon.gamma
