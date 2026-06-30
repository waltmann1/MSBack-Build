import pickle as pkl
from msback.egnn_utils import *
from time import time
import numpy as np
import os.path
import pandas as pd
import re
import yaml

def timer(func):
    def time_wrapper(*args, **kwargs):
        now = time()
        result = func(*args, **kwargs)
        end = time()
        total = end - now
        print(f"The total time was {total} seconds.")
        return result

    return time_wrapper

def launch_pool_in_batches(pool, n_threads):
    n_batches = int(np.ceil(len(pool) / n_threads))
    for batch in range(n_batches):
        start = batch * n_threads
        end = np.min( [(batch + 1) * n_threads, len(pool)])
        subpool = pool[start:end]
        for thread in subpool:
            thread.start()
        for thread in subpool:
            thread.join()

def pro_res_to_ohe(string_list):

    amino_acids = {
    "ALA": 1,  # Alanine
    "ARG": 2,  # Arginine
    "ASN": 3,  # Asparagine
    "ASP": 4,  # Aspartic acid
    "CYS": 5,  # Cysteine
    "GLU": 6,  # Glutamic acid
    "GLN": 7,  # Glutamine
    "GLY": 8,  # Glycine
    "HIS": 9,  # Histidine
    "ILE": 10, # Isoleucine
    "LEU": 11, # Leucine
    "LYS": 12, # Lysine
    "MET": 13, # Methionine
    "PHE": 14, # Phenylalanine
    "PRO": 15, # Proline
    "SER": 16, # Serine
    "THR": 17, # Threonine
    "TRP": 18, # Tryptophan
    "TYR": 19, # Tyrosine
    "VAL": 20  # Valine
    }

    indices = [amino_acids[string] for string in string_list]
    return np.array(indices)

def pro_allatom_to_ohe(string_list):
    '''List all possible atom names'''

    atom_types = {
        'C':1,
        'CA':2,
        'CB':3,
        'CD':4,
        'CD1':5,
        'CD2':6,
        'CE':7,
        'CE1':8,
        'CE2':9,
        'CE3':10,
        'CG':11,
        'CG1':12,
        'CG2':13,
        'CH2':14,
        'CZ':15,
        'CZ2':16,
        'CZ3':17,
        'N':18,
        'ND1':19,
        'ND2':20,
        'NE':21,
        'NE1':22,
        'NE2':23,
        'NH1':24,
        'NH2':25,
        'NZ':26,
        'O':27,
        'OT2':27,
        'OD1':28,
        'OD2':29,
        'OE1':30,
        'OE2':31,
        'OG':32,
        'OG1':33,
        'OH':34,
        'SD':35,
        'SG':36
    }

    indices = [atom_types[string] for string in string_list]
    return np.array(indices)

def pro_atom_to_ohe(string_list):

    atoms = {
            'C': 1,
            'O': 2,
            'N': 3,
            'S': 4,
        }
    indices = [atoms[string[0]] for string in string_list]
    return np.array(indices)

def read_rtp_file(file_path):
    with open(file_path, 'r') as file:
        rtp_data = file.readlines()  # Read the file as a list of lines
    return rtp_data


def parse_rtp(rtp_data):
    rtp_dict = {}
    residue_type = None
    atom_dict = {}
    
    # Process each line in the RTP data
    for line in rtp_data:
        line = line.strip()
        
        # If a new residue type is found, initialize a new dictionary for it
        residue_match = re.match(r"\[\s*[A-Z0-9]*\s*\]", line)
        if residue_match:
            if residue_type:  # If there is an existing residue, save it before moving on
                rtp_dict[residue_type] = atom_dict
            
            residue_type = residue_match.group(0)[2:-2]
            atom_dict = {}
            reading_atoms = True  # Start reading atoms
        # Process atoms section (inside [ atoms ] block)
        elif residue_type and re.match(r"\[ atoms \]", line):
            # Start reading atoms after "[ atoms ]" section header
            continue
        elif re.match(r"\[ bonds \]", line) or re.match(r"\[ impropers \]", line) or re.match(r"\[ cmap \]", line) or re.match(r"\[ bondedtypes \]", line):
            # Stop reading when the "[ bonds ]" section is reached
            reading_atoms = False
            continue
        elif residue_type and reading_atoms:
            # Parse atom data line
            parts = line.split()
            if len(parts) >= 2:
                atom_name = parts[0]
                atom_type = parts[1]
                if atom_name.isupper() and atom_type.isupper():
                    atom_dict[atom_name] = atom_type

    # Don't forget to add the last residue type to the dictionary
    if residue_type:
        rtp_dict[residue_type] = atom_dict
    
    return rtp_dict

def get_ff_data(rtp_path=os.path.dirname(os.path.abspath(__file__))+'/Data/aminoacids.rtp', nb_path=os.path.dirname(os.path.abspath(__file__))+'/Data/ffnonbonded.csv', bond_path=os.path.dirname(os.path.abspath(__file__))+'/Data/bondtypes.csv'):
    rtp_lines = read_rtp_file(rtp_path)
    rtp_data = parse_rtp(rtp_lines)
    lj_data = pd.read_csv(nb_path, sep='\t', comment=';')
    bond_data = pd.read_csv(bond_path, sep='\t', comment=';')
    return rtp_data, lj_data, bond_data

def load_model(model_path, ckp, device): #, sym='e3', pos_cos=None, seq_feats=0, seq_decay=100):
    '''Load model from a given path to device'''

    # load hyperparams associated with specific model
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/" +  model_path
    model_params = pkl.load(open(f'{model_path}/params.pkl', 'rb'))

    ## TODO -- get rid or these inputs and always set pos_cos, num_positions to None
    pos_cos = model_params['pos_cos']

    # don't use normal pos encoding if using sin/cos
    if pos_cos > 0.001:
        num_positions = None
    else:
        pos_cos = None
        num_positions = None  # change this back ifthe
        #num_positions = model_params['max_atoms']

    model = EGNN_Network_time(
        num_tokens = model_params['res_dim'],
        atom_dim = model_params['atom_dim'],
        num_positions = num_positions, #model_params['max_atoms'],
        dim = model_params['dim'],
        depth = model_params['depth'],
        num_nearest_neighbors = model_params['num_nearest_neighbors'],
        global_linear_attn_every = 0,
        coor_weights_clamp_value = 2.,
        m_dim=model_params['mdim'],
        fourier_features = 4,
        time_dim=0,
        res_dim=model_params['res_dim'],
        sym=model_params['sym'],
        emb_cos_scale=pos_cos,
        seq_feats=model_params['seq_feats'],
        seq_decay=model_params['seq_decay'],
        act=model_params['act'],
    ).to(device)

    # load model
    state_dict_path = f'{model_path}/state-{ckp}.pth'
    model.load_state_dict(torch.load(state_dict_path))

    return model

#@ optionally add these functions to eval_utils
def load_model_adj(model_path, ckp, device, config=None): #, sym='e3', pos_cos=None, seq_feats=0, seq_decay=100):
    '''Load model from a given path to device'''
    
    # load hyperparams associated with specific model
    # model_params = pkl.load(open(f'{model_path}/params.pkl', 'rb')) 
    # Load configuration from YAML file

    model_path = os.path.dirname(os.path.abspath(__file__)) + "/" + model_path
    with open(f'{model_path}/config.yaml', 'r') as file:
        model_params = yaml.safe_load(file)
    ## TODO -- get rid or these inputs and always set pos_cos, num_positions to None
    pos_cos = model_params['pos_cos']

    # don't use normal pos encoding if using sin/cos
    if pos_cos > 0.001:
        num_positions = None
    else: 
        pos_cos = None
        num_positions = None  # change this back ifthe 
        #num_positions = model_params['max_atoms']

    import msback.adjoint_egnn_utils as adj
    model = adj.EGNN_Network_time(
        num_tokens = model_params['res_dim'],
        atom_dim = model_params['atom_dim'],
        num_positions = num_positions, #model_params['max_atoms'],
        dim = model_params['dim'],               
        depth = model_params['depth'],
        num_nearest_neighbors = model_params['nneigh'],
        global_linear_attn_every = 0,
        coor_weights_clamp_value = 2.,  
        m_dim=model_params['mdim'],
        fourier_features = 4, 
        time_dim=0,
        res_dim=model_params['res_dim'],
        sym=model_params['sym'],
        emb_cos_scale=pos_cos,
        seq_feats=model_params['seq_feats'],
        seq_decay=model_params['seq_decay'],
        act=model_params['act'],
    ).to(device)

    # load model 
    state_dict_path = f'{model_path}/state-{ckp}.pth' 

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(state_dict_path))

    return model

def get_ca_pos(xyz, aa_to_cg):
    '''Ca positions of every atom'''
    xyz_ca = []
    for xyz_ref, map_ref in zip(xyz, aa_to_cg):
        print(xyz_ref)
        print(map_ref)
        xyz_ca_i = xyz_ref[map_ref]
        xyz_ca.append(xyz_ca_i)

    return np.array(xyz_ca)
