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
AAname = "mlv_full_hex_withIP6.pdb"

ip6 = "YZABCD"

trimers = ["ROM", "SBQ", "FAC", "GJE", "KIN"]


trimers = ["GIK", "DFH", "ACE", "PBR", "QMO", "JLN"]

aa = AAProtein(AAname)

flos = [cp.deepcopy(aa) for _ in range(6)]
for i in range(6):
    flos[i].u.atoms = flos[i].u.atoms[(flos[i].u.atoms['resname']=="IP6") & (flos[i].u.atoms['chain']==ip6[i])]
    flos[i].write("IP6_" + str(i) + ".pdb")

AAname = "6HWX.pdb"
for i in range(len(trimers)):
    aa = AAProtein(AAname)
    aa.u.atoms = aa.u.atoms[(aa.u.atoms['name']=='CA') & (aa.u.atoms['chain'].apply(lambda x: str(x) in trimers[i]))]


    start = time.time()

    aa.add_backbone_sidechains()
    flo = aa.flowback_adjoint_protein()
    end = time.time()
    flo.add_C_terminal_O()
    flo.write("flowed_"+ str(i)+ ".pdb")
    temp_flows = [cp.deepcopy(flo) for _ in range(3)]
    for x in range(3):
        temp_flows[x].u.atoms = temp_flows[x].u.atoms[(temp_flows[x].u.atoms['chain']==trimers[i][x])]
    flos.extend(temp_flows)
    print(len(flos))


flos[0].write("IP6.pdb")
IP6 = IP6("IP6.pdb")
ff_add= IP6.write_ff()
total = MSToolProteinComplex(flos[6:],ligands=flos[:6])
total.write("total_nornem.pdb",  ligands=True)
quit()
total.run_rnem(freeze_backbone=True, ff_add=ff_add)
quit()

total.write("total.pdb")

total = MSToolProteinComplex(flos[:-6])

total.write("total_noIP6.pdb")


