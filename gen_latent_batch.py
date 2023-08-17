import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
from bo import sascorer
from torch.utils.data import Dataset, DataLoader
import numpy as np  
from fast_jtnn import *
import rdkit
from fast_jtnn.mol_tree import Vocab, MolTree
from fast_jtnn.jtnn_vae import JTNNVAE
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.jtmpn import JTMPN
from fast_jtnn.mpn import MPN
from fast_jtnn.nnutils import create_var
from fast_jtnn.datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

parser = argparse.ArgumentParser()

parser.add_argument("--vocab",required=True) 

parser.add_argument("--smiles",required=True) 
parser.add_argument("--model", required=True)
parser.add_argument("--hidden",  type=int,default=450)
parser.add_argument( "--latent", type=int, default=56)
parser.add_argument("--depth", type=int,default=20)
args = parser.parse_args()
print(args)
import os
import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import pickle

from fast_jtnn import *
import rdkit
from fast_jtnn.mol_tree import Vocab, MolTree
from fast_jtnn.jtnn_vae import JTNNVAE
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.jtmpn import JTMPN
from fast_jtnn.mpn import MPN
from fast_jtnn.nnutils import create_var
from fast_jtnn.datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    mol_tree=[[mol_tree]]
    dataset = MolTreeDataset(mol_tree, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
    for b in dataloader:
        mol_vec=model.encode_latent(b)
        mol_vec=mol_vec.tolist()
        print(smiles.strip(),"\t",mol_vec[0])
    return [smiles,mol_vec]

   # lg = rdkit.RDLogger.logger() 
   # lg.setLevel(rdkit.RDLogger.CRITICAL)

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)
smile_path=args.smiles
batch_size = 1
hidden_size = int(args.hidden)
latent_size = int(args.latent)
depth = int(args.depth)

model = JTNNVAE(vocab, hidden_size, latent_size, 20,3)
print(model)
model.load_state_dict(torch.load(args.model))
model = model.cuda()

print(smile_path)
with open(smile_path,"r") as f:
    smiles = f.readlines()
data=[i for i in smiles if i!="nan"]
parser = OptionParser()

smiles_list=[]
mol_trees=[]
print("preprocessing smiles of length: %i"%len(data))
smiles_list=[]
mol_trees=[]
#pool.map(tensorize,data)

for i in data:
    try:
        tensorize(i)
    except:
        print(i+"\tnan")
#for smi,vec in pool.map(tensorize,data):
#    print(smi,"\t",vec)
  #  for tree,smile in pool.map(tensorize,data):
  #      smiles_list.append(smile)
  #      mol_trees.append(tree)
  #  with open("latent/smiles_%s.txt"%data_i,"w") as ft:
  #      for i in smiles_list:
  #          ft.write(i+"\n")
  #  with open("data/spi_pkl/tensor00.pkl", 'wb') as f:
  #      pickle.dump(mol_trees, f, pickle.HIGHEST_PROTOCOL)
  #  smiles=smiles_list
  #  for i in range(len(smiles)):
  #      smiles[ i ] = smiles[ i ].strip()
  #  smiles_rdkit = []
  #  for i in range(len(smiles)):
  #      smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[ i ]), isomericSmiles=True))

 #   logP_values = []
 #   for i in range(len(smiles)):
 #       logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))

  #  import networkx as nx
  #  latent_points = []
  # # loader = MolTreeFolder(args.train, vocab, batch_size,shuffle=False, num_workers=4)
  #  for batch in loader:
  #      mol_vec = model.encode_latent(batch)
  #      latent_points.append(mol_vec.data.cpu().numpy())

  #  # We store the results
  #  latent_points = np.vstack(latent_points)
  #  np.savetxt('latent/latent_%s.txt'%data_i, latent_points)

