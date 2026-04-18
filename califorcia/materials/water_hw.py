from ..models import LorentzModel, DebyeModel, CombinedModel

# Hough and White (1980) water model
# Ref: Adv. Colloid Interface Sci. 14, 3 (1980)

# 1. Ultraviolet oscillator
# eps_uv - 1 = 0.778, omega_uv = 1.904e16 rad/s
c_uv = 0.778
w_uv = 1.904e16
lorentz_uv = LorentzModel(c_uv, w_uv, 0.0)

# 2. Infrared oscillator
# eps_ir - eps_uv = 3.062, omega_ir = 5.48e13 rad/s
c_ir = 3.062
w_ir = 5.48e13
lorentz_ir = LorentzModel(c_ir, w_ir, 0.0)

# 3. Microwave (Debye) relaxation
# eps_static - eps_ir = 73.48, tau = 1e-11 s
eps_0 = 78.32
eps_ir_limit = 4.84
tau = 1.0e-11

# DebyeModel(D, I, tau) returns I + (D - I) / (1 + xi*tau)
# We set I = 1.0 and D = 1.0 + (eps_0 - eps_ir_limit)
debye_mw = DebyeModel(1.0 + (eps_0 - eps_ir_limit), 1.0, tau)

# Combine models: 1 + sum(L_i - 1) + (D - 1)
# = 1 + (1+c_uv-1) + (1+c_ir-1) + (1+(78.32-4.84)-1)
# = 1 + 0.778 + 3.062 + 73.48 = 78.32 (at xi=0)
epsilon = CombinedModel([lorentz_uv, lorentz_ir, debye_mw])
materialclass = epsilon.materialclass
