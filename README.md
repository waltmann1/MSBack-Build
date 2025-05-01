# msback
Tools for backmapping from coarse resolutions

msback includes the work of many people please cite the following works:

Waltmann C, Wang Y, Yang C, Kim S, Voth G. MSBack: Multiscale Backmapping of Highly Coarse-grained Proteins Using Constrained Diffusion. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-04scq (update when published)

Siyoung Kim
The Journal of Physical Chemistry B 2023 127 (49), 10488-10497
DOI: 10.1021/acs.jpcb.3c05593 

Ingraham, J.B., Baranov, M., Costello, Z. et al. Illuminating protein space with a programmable generative model. Nature 623, 1070–1078 (2023). https://doi.org/10.1038/s41586-023-06728-8

Michael S. Jones, Smayan Khanna, and Andrew L. Ferguson
Journal of Chemical Information and Modeling 2025 65 (2), 672-692
DOI: 10.1021/acs.jcim.4c02046

Yuxing Peng, Alexander J. Pak, Aleksander E. P. Durumeric, Patrick G. Sahrmann, Sriramvignesh Mani, Jaehyeok Jin, Timothy D. Loose, Jeriann Beiter, and Gregory A. Voth
The Journal of Physical Chemistry B 2023 127 (40), 8537-8550
DOI: 10.1021/acs.jpcb.3c04473 


to install this one must first install openmscg with mstool
then download and install the version of chroma from https://github.com/waltmann1/MyChroma
using the README there you will be able to download the weights for the chroma model and register your use
the ".pt" weight files should be placed in a folder named chroma_weights/ in this directory
	-otherwise modify the get_local_weights function in /src/msback/MSToolProtein.py
after additional dependencies are installed
install msback via "pip install -e ." (in this directory)

this one suggested way to install the necessary packages

conda create -n msback python=3.9
conda activate msback
conda install numpy cython scipy matplotlib 
conda install openmm
conda install mdanalysis
pip install mscg-mstool/ (install mscg-mstool, need to be in the proper directory)
pip install mdtraj
pip install -e chroma/ (run this after cloning from github again in the proper directory)
pip install egnn.pytorch
pip install sidechainnet
pip install rdkit
pip install -e msback/

if a package named the_package fails when you try to run, pip uninstall the_package and pip install the_package

see mback/src/msback/examples for usage examples


