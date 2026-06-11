import numpy as np
import mstool
import sys
from msback.MSToolProtein import MSToolProtein
from msback.MSToolProtein import MSToolProteinComplex
from msback.MSToolProtein import AAProtein
from msback.Simulation import QuickSim
import time
import copy as cp
import time
import torch
import chroma.utility.polyseq as polyseq
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.stats

def get_seq(s0):

    seq = []
    for ind in range(len(s0)):
        r0 = polyseq.index_to_single(s0[ind])
        seq.append(r0)
    return seq

def seq_compare(s0, s1):
    diff = np.subtract(s0,s1) != 0
    mutations = []
    for ind,thing in enumerate(diff):
        if diff[ind]:
            r0 = polyseq.index_to_single(s0[ind])
            r1 = polyseq.index_to_single(s1[ind])
            sring = ""
            sring += r0
            sring +=str(ind+1)
            sring += r1
            mutations.append(sring)
    return mutations

def get_seq_code(ref, mut):
    
    code = cp.deepcopy(ref)
    for m in mut:
        index = int(m[1:-1]) - 1
        first = polyseq.res_to_index(m[0])
        if first != code[index]:
            print("error with mutation ", mut, " start is ", polyseq.index_to_single(code[index]))
        code[index] = polyseq.res_to_index(m[-1])
    return code


ref = np.load("results_one/sequence" + str(0) + ".npy")

seq = np.load("results_one/sequence.npy")
diff = np.load("results_one/energy_diff.npy")

matrix = np.zeros((20, 200))

for ind, d in enumerate(diff):
    s = seq[ind]
    mut = seq_compare(ref, s)
    mut = mut[0]
    if len(mut) != 0:
        resid_index = int(mut[1:-1]) - 1
        aa_index = polyseq.res_to_index(mut[-1])
        matrix[19 - aa_index][resid_index + 3] = d
        if resid_index + 3 == 3 and aa_index ==17 :
            print(mut)



plt.figure(figsize=(10,5))
plt.imshow(matrix[:, 111:], cmap='bwr', vmin=-5, vmax=5, extent= [1, 90, 0 , 20], aspect='auto')
plt.xlabel("Resid")
plt.ylabel("Amino Acid")

words = [polyseq.index_to_single(i) for i in range(20)]
plt.yticks([.5 + i for i in range(20)], words)
plt.tick_params(axis='y', length=0)

plt.colorbar(label='$\Delta U_{predicted}$')
plt.savefig("fig_one_matrix_barstart.png")


plt.show()






