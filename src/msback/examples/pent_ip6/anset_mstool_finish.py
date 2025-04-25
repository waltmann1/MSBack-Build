import numpy as np
#import mstool
import sys
from msback.MSToolProtein import MSToolProtein
from msback.MSToolProtein import MSToolProteinComplex
from msback.MSToolProtein import AAProtein
from msback.Simulation import QuickSim
from msback.Ligand import IP6
import threading
import time
import msback.Utils as u
import torch
def memory_stats():
    print("memory stats")
    for i in range(torch.cuda.device_count()):
        print(i)
        print(torch.cuda.memory_allocated(i) * 1024**2)
        print(torch.cuda.memory_reserved(i) * 1024**2)

AAnames = ["hard_alignedcapsid" + str(i) + ".pdb" for i in range(5)]
yaml_name = "2m8l_CG.yaml"

diffused_proteins = [AAProtein(AAnames[i], yaml = yaml_name) for i in range(len(AAnames))]


lizt= diffused_proteins

lizt2 = []
ref = MSToolProtein("ip6_cg0_clean.pdb")
n_ip6 = len(ref.u.atoms.name.values)
ip6 = IP6("LIG.pdb")
ff_add = ip6.write_ff()
lizt2 = [ip6 for _ in range(n_ip6)]
for ind in range(n_ip6):
    lizt2[ind].shift(ref.u.atoms[['x', 'y', 'z']].values[ind])

hept_complex = MSToolProteinComplex(lizt, ligands=lizt2)
name = hept_complex.run_rnem(ff_add=ff_add)

pro = MSToolProtein(name)

pro.write("rnem.pdb")

