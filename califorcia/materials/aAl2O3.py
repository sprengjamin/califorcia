from ..models import SellmeierModel

B = [1.024, 1.058, 5.28]
C = [(0.0614e-6)**2, (0.111e-6)**2, (17.93e-6)**2]

epsilon = SellmeierModel(B, C)
materialclass = epsilon.materialclass
