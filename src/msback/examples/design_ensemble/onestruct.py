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

AAname = "run_BB_reindex.pdb"

#AAname = "state_A.pdb"
a =time.time()
aa = MSToolProtein(AAname)
b= time.time()

sequence, mutations, gradients = aa.get_chroma_energy()
print(mutations.shape, gradients.shape)
nbins = gradients.shape[0]
 

result_dir = "results_one/"

np.save(result_dir + "sequence" + str(0) + ".npy", sequence)
np.save(result_dir + "energy_diff.npy", gradients)
np.save(result_dir + "sequence.npy", mutations)

