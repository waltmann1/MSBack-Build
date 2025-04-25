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

def diffusion_worker(aa, cg_protein, chroma, device, diffused_proteins):
    cg_protein.u.atoms = cg_protein.u.atoms[:130]
    rmsd, mstp = aa.get_protein_aligned_with_cg_group(cg_protein)
    name = "aligned" + aa.name + ".pdb" 
    mstp.write(name)
    mstp_aligned_cg = AAProtein(name, yaml="2m8l_CG.yaml").diffuse_to_CG(cg_protein,output_soft=True, pack_sidechains=True, skip_hard=False, chroma=chroma, device=device)
    diffused_proteins.append(mstp_aligned_cg)

memory_stats()
AAnames = ["capsid" + str(i) + ".pdb" for i in range(5)]
yaml_name = "2m8l_CG.yaml"
CGnames = ["cg" + str(i)+ ".pdb" for i in range(5)]

aas = [AAProtein(AAnames[i], yaml = yaml_name) for i in range(len(CGnames))]
cg_proteins = [MSToolProtein(CGname) for CGname in CGnames]

n_threads = torch.cuda.device_count()

chromas = [MSToolProtein("cg0.pdb").load_chroma(device="cuda:" + str(i)) for i in range(n_threads)]
memory_stats()

#quit()
diffused_proteins = []
pool = [threading.Thread(target=diffusion_worker, args=(aas[i], cg_proteins[i], chromas[i%n_threads][0], chromas[i%n_threads][1], diffused_proteins)) for i,_ in enumerate(cg_proteins)]
memory_stats()
u.launch_pool_in_batches(pool, n_threads)
memory_stats()


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


