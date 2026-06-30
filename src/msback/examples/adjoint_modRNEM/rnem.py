import numpy as np
import mstool
import sys
import copy as cp
from msback.MSToolProtein import MSToolProtein
from msback.MSToolProtein import MSToolProteinComplex
from msback.MSToolProtein import AAProtein
from msback.Simulation import QuickSim
from msback.Ligand import IP6
import time
AAname = "total_nornem.pdb"

ip6 = "STUVWX"

trimers = ["ROM", "SBQ", "FAC", "GJE", "KIN"]


trimers = ["GIK", "DFH", "ACE", "PBR", "QMO", "JLN"]

aa = AAProtein(AAname)

flos = [cp.deepcopy(aa) for _ in range(24)]
chains = list("ABCDEFGHIJKLMONPQRSTU")[:18]
for i in range(18):
    flos[i].u.atoms = flos[i].u.atoms[(flos[i].u.atoms['chain']==chains[i])]
for i in range(18, 24):
    flos[i].u.atoms = flos[i].u.atoms[(flos[i].u.atoms['resname']=="IP6") & (flos[i].u.atoms['chain']==ip6[i-18])]



flos[-1].write("IP6.pdb")

IP6 = IP6("IP6.pdb")
ff_add= IP6.write_ff()
total = MSToolProteinComplex(flos[:-6],ligands=flos[-6:])
total.write("check.pdb", ligands=True)

energies = [1e10]
names = ["1e10"]

#energies =  [1e5]
#names = ["1e5"]

for ind, thing in enumerate(energies):
    name = total.run_rnem(keep_heavy=True, ff_add=ff_add, energy=thing)
    al = MSToolProtein(name)
    al.write("rnem_heavy_" + str(names[ind])+ ".pdb")

