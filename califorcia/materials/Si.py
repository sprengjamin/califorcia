from ..models import LorentzModel

c1 = 7.666
w1 = 3.685
c2 = 2.913
w2 = 6.354
c3 = 0.06169
w3 = 48.66

f_list = [c1, c2, c3]
w_list = [w1, w2, w3]
g_list = [0.0, 0.0, 0.0]

epsilon = LorentzModel(f_list, w_list, g_list)
materialclass = epsilon.materialclass
