from ..models import SellmeierModel

B = [0.696166300, 0.407942600, 0.897479400]
C = [4.67914826e-3*1.e-12, 1.35120631e-2*1.e-12, 97.9325*1.e-12]

epsilon = SellmeierModel(B, C)
materialclass = epsilon.materialclass
