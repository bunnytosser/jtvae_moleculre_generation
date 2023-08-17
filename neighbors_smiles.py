import pickle
import gzip
from bo.sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import os.path
import numpy as np
import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
import argparse
import torch
import torch.nn as nn

from fast_jtnn.mol_tree import Vocab, MolTree
from fast_jtnn.jtnn_vae import JTNNVAE
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.jtmpn import JTMPN
from fast_jtnn.mpn import MPN
from fast_jtnn.nnutils import create_var
from fast_jtnn.datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

# We define the functions used to load and save objects
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret
parser = argparse.ArgumentParser()
parser.add_argument("--vocab",required=True) 
parser.add_argument("--model")
parser.add_argument("--hidden",  type=int,default=450)
parser.add_argument( "--latent", type=int, default=56)
parser.add_argument("--depth", type=int,default=20)
args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

hidden_size = int(args.hidden)
latent_size = int(args.latent)
depth = int(args.depth)

model = JTNNVAE(vocab, hidden_size, latent_size, 20,3)
model.load_state_dict(torch.load(args.model))
model = model.cuda()

# We load the random seed

import pandas as pd
data= np.loadtxt("predict/latent_features.txt")
data=data[:3,:]
print(data)
for i in np.arange(len(data)):
    if i<=1: continue
    valid_smiles = []
    new_features = []

    for p in np.arange(200):
        noise = np.random.normal(0,0.1,56).reshape(56,)
        added=noise+data[i,:]
        all_vec = added.reshape((1,-1))
        tree_vec,mol_vec = np.hsplit(all_vec, 2)
        tree_vec = create_var(torch.from_numpy(tree_vec).float())

        mol_vec = create_var(torch.from_numpy(mol_vec).float())
        s = model.decode(tree_vec, mol_vec, prob_decode=False)
        if s is not None: 
            valid_smiles.append(s)
            print(s)
            new_features.append(all_vec)
    with open("noisy_smiles_{}.txt".format(i),"w") as f:
        for k in valid_smiles:
            f.write(k+"\n")

