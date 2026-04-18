from ..models import LorentzModel

wj = 1.911e16 # 12.6 eV/hbar
cj = 1.282

epsilon = LorentzModel(cj, wj, 0.0)
materialclass = epsilon.materialclass
