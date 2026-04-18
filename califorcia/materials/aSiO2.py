from ..models import SellmeierModel

B = [0.696, 0.408, 0.897]
C = [(0.0684e-6)**2, (0.116e-6)**2, (9.896e-6)**2]

epsilon = SellmeierModel(B, C)
materialclass = epsilon.materialclass
