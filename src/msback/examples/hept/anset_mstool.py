import numpy as np
import mstool
import sys
from msback.MSToolProtein import MSToolProtein
from msback.MSToolProtein import MSToolProteinComplex
from msback.MSToolProtein import AAProtein
from msback.Simulation import QuickSim
import time

AAname = "hept_config_pent2_2.pdb"
yaml_name = "2m8l_CG.yaml"
CGnames = ["cg" + str(i)+ ".pdb" for i in range(7)]

aa = AAProtein(AAname, yaml = yaml_name)
cg_proteins = [MSToolProtein(CGname) for CGname in CGnames]


diffused_proteins = []
#ALIGNMENT
for cg_protein in cg_proteins:
    cg_protein.u.atoms = cg_protein.u.atoms[:130]
    rmsd, mstp = aa.get_protein_aligned_with_cg_group(cg_protein)
    print(rmsd)
    print(mstp.get_cg_protein().rmsd(cg_protein))
    mstp.write("check_aa.pdb")
    #cg_protein.write("check_cg.pdb")
    new_aa=AAProtein("check_aa.pdb", yaml = yaml_name)
    mstp_aligned_cg = new_aa.diffuse_to_CG(cg_protein,output_soft=True, pack_sidechains=True, skip_hard=False)
    diffused_proteins.append(mstp_aligned_cg)

hept_complex = MSToolProteinComplex(diffused_proteins)
hept_complex.write("hept.pdb")

heptamer = MSToolProtein("hept.pdb")
flowed = heptamer.flowback_protein()
flowed_name = "finish_flowed.pdb"
flowed.write(flowed_name)
qs = QuickSim(flowed_name)
qs.run_min(max_iterations=100)
new_name = flowed_name.split(".")[0] + "_em.pdb"

new_prot=AAProtein(new_name, yaml=yaml_name)
new_prot.resmap = aa.resmap
new_prot.u.atoms = new_prot.u.atoms[new_prot.u.atoms.name=="CA"]

old_prot = AAProtein(flowed_name, yaml=yaml_name)
old_prot.resmap = aa.resmap
old_prot.u.atoms = old_prot.u.atoms[old_prot.u.atoms.name=="CA"]

#aa_prot.u.atoms = aa_prot.u.atoms[aa_prot.u.atoms.name=="CA"]

rmsd = new_prot.rmsd(old_prot)
print("rmsd due to em", rmsd)


new_prot.get_cg_protein(use_resmap=True).write("cg_flowed_em.pdb")

cg_rmsd = new_prot.rmsd_cg(cg_protein, use_resmap=True)
print("RMSD on the CG level", cg_rmsd)



cg_rmsd = old_prot.rmsd_cg(cg_protein, use_resmap=True)
print("RMSD on the CG level before EM", cg_rmsd)


