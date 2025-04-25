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
def memory_stats():
    print("memory stats")
    for i in range(torch.cuda.device_count()):
        print(i)
        print(torch.cuda.memory_allocated(i) * 1024**2)
        print(torch.cuda.memory_reserved(i) * 1024**2)

def diffusion_worker(aa, cg_protein, chroma, device, diffused_proteins):
    cg_protein.u.atoms = cg_protein.u.atoms[:130]
    rmsd, mstp = aa.get_protein_aligned_with_cg_group(cg_protein)
    name = "check_aa" + str(device)+".pdb"
    mstp.write(name)
    #cg_protein.write("check_cg.pdb")
    print("info")
    print(rmsd)
    mstp_aligned_cg = AAProtein(name, yaml="2m8l_CG.yaml").diffuse_to_CG(cg_protein,output_soft=True, pack_sidechains=True, skip_hard=False, chroma=chroma, device=device)
    diffused_proteins.append(mstp_aligned_cg)

memory_stats()
AAnames =["hept_config_pent2_" + str(2) + ".pdb" for i in range(7)]
yaml_name = "2m8l_CG.yaml"
CGnames = ["cg" + str(i)+ ".pdb" for i in range(7)]

#"""
aas = [AAProtein(AAname, yaml = yaml_name) for AAname in AAnames]
cg_proteins = [MSToolProtein(CGname) for CGname in CGnames]

n_threads = torch.cuda.device_count()
chromas = [MSToolProtein("../cg0.pdb").load_chroma(device="cuda:" + str(i)) for i in range(n_threads)]
#chromas = [(None, None) for _ in range(2)]
memory_stats()

#quit()
diffused_proteins = []
pool = [threading.Thread(target=diffusion_worker, args=(aas[i], cg_proteins[i], chromas[i%n_threads][0], chromas[i%n_threads][1], diffused_proteins)) for i,_ in enumerate(cg_proteins)]
memory_stats()
u.launch_pool_in_batches(pool, n_threads)
memory_stats()

hept_complex = MSToolProteinComplex(diffused_proteins)
#hept_complex.write("hept.dms")
hept_complex.write("hept.pdb")

cg_complex = MSToolProteinComplex(cg_proteins)
#cg_complex.write("hept_cg.dms")
cg_complex.write("hept_cg.pdb")

#"""
cg_heptamer = MSToolProtein("hept_cg.pdb")
heptamer = MSToolProtein("hept.pdb")

flowed = heptamer.flowback_protein()
flowed_name = "finish_flowed.pdb"
flowed.write(flowed_name)
qs = QuickSim(flowed_name)
qs.run_min(max_iterations=100)
new_name = flowed_name.split(".")[0] + "_em.pdb"

new_prot=AAProtein(new_name, yaml=yaml_name)
new_prot.u.atoms = new_prot.u.atoms[new_prot.u.atoms.name=="CA"]

old_prot = AAProtein(flowed_name, yaml=yaml_name)
old_prot.u.atoms = old_prot.u.atoms[old_prot.u.atoms.name=="CA"]


rmsd = new_prot.rmsd(old_prot)
print("rmsd due to em", rmsd)

memory_stats()


