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

n_mut = 3743


ref = np.load("results/sequence" + str(0) + ".npy")

mutants = []
gradients = []
seqs = []
ref_fe = np.load("pmf_kt.npy")[2:-2]

bin_weights = np.exp(- ref_fe) / np.sum(np.exp(-ref_fe))
bin_centers =np.load("bin_centers.npy")[2:-2]

old_diff = np.min(ref_fe) - np.max(ref_fe)
#quit()
for i in range(1, n_mut+1):
    seq = np.load("results/sequence" + str(i) + ".npy")
    ref_grad = np.load("results/gma_" + str(i) + ".npy")[2:-2]
    ref_grad_average = np.dot(ref_grad, bin_weights)
    term2 = np.subtract(ref_grad, ref_grad_average)
    if i %1000 ==0:
        plt.plot(bin_centers, term2)
        #plt.plot(bin_centers, ref_fe)
        plt.show()
    new_fe = np.add(ref_fe, term2)
    new_diff = np.min(new_fe) - np.max(new_fe)
    mut = seq_compare(ref, seq)
    if i%1000==0:
        print(mut, old_diff - new_diff, old_diff, new_diff)

    seqs.append(seq)
    mutants.append(mut)
    gradients.append(old_diff - new_diff)


gradients = np.array(gradients)
seqs = np.array(seqs)


idx = np.argsort(gradients)

sorted_grads = gradients[idx]
sorted_seqs = seqs[idx]
np.save("results_grad/diff.npy", sorted_grads)
np.save("results_grad/sequences.npy", sorted_seqs)


best = max(np.multiply(-1, gradients))
print( best)
index = gradients.index(-best)
print(index, mutants[index])
#print(max(np.multiply(-1, gradients)))
#print(np.max(gradients), gradients.index(max(gradients)), mut[gradients.index(max(gradients))])
