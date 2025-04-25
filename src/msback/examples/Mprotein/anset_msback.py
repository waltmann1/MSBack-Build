import numpy as np
import mstool
import sys
from msback.MSToolProtein import MSToolProtein
from msback.MSToolProtein import MSToolProteinComplex
from msback.MSToolProtein import AAProtein
from msback.Simulation import QuickSim
import time

AAname = "M_long_AA.pdb"
yaml_name = "M_mapping.yaml"
CGname = "M_CG.pdb"


aa = AAProtein(AAname, yaml = yaml_name)
cg_protein = MSToolProtein(CGname)

#ALIGNMENT
rmsd, mstp = aa.get_protein_aligned_with_cg_group(cg_protein)
#print(rmsd)
#print(mstp.rmsd(aa))
#writing checks
#mstp = MSToolProtein("aligned.pdb")
#aligned_cg = AAProtein("aligned.pdb", yaml="M_mapping.yaml").diffuse_to_CG(cg_protein)

mstp_aligned_cg = mstp.diffuse_to_CG(cg_protein, pack_sidechains=True, skip_hard=False)


flowed = mstp_aligned_cg.flowback_protein()
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


