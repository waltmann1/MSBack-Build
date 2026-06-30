import numpy as np
import mstool
import sys
from msback.MSToolProtein import MSToolProtein
from msback.MSToolProtein import MSToolProteinComplex
from msback.MSToolProtein import AAProtein
from msback.Simulation import QuickSim
import threading
import time
import msback.Utils as u
import torch
import copy as cp

AAname = "rnem_heavy_1e10_back1e4.pdb"

aa = AAProtein(AAname)

flos = [cp.deepcopy(aa) for _ in range(18)]
chains = list("ABCDEFGHIJKLMONPQRSTU")[:18]
for i in range(18):
    flos[i].u.atoms = flos[i].u.atoms[(flos[i].u.atoms['chain']==chains[i])]

total = MSToolProteinComplex(flos)

qs_name = AAname[:-4]+ "_noIP6.pdb"
total.write(qs_name)



qs = QuickSim(qs_name)
qs.run_min(max_iterations=10000)
new_name = qs_name.split(".")[0] + "_em.pdb"

new_prot=AAProtein(new_name)
new_prot.u.atoms = new_prot.u.atoms[new_prot.u.atoms.name=="CA"]

old_prot = AAProtein(qs_name)
old_prot.u.atoms = old_prot.u.atoms[old_prot.u.atoms.name=="CA"]


rmsd = new_prot.rmsd(old_prot)
print("rmsd due to em", rmsd)
