import pickle
import gzip
from bo.sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import os.path
from rdkit import Chem
import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
import argparse
import torch
import torch.nn as nn
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from fast_jtnn.mol_tree import Vocab, MolTree
from fast_jtnn.jtnn_vae import JTNNVAE
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.jtmpn import JTMPN
from fast_jtnn.mpn import MPN
from fast_jtnn.nnutils import create_var
from fast_jtnn.datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
target_smiles="O=P(O)(O)OC[C@@H]1[C@@H](O)[C@@H](O)[C@@H](O1)N2C=NC3=C2N=CN=C3NC4=CC=CC(O)=C4"
maccs_fp_target=MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(target_smiles))

target_smiles="n1cnc2c(c1Nc1cc(ccc1)O)ncn2[C@H]1[C@@H]([C@@H]([C@H](O1)CO)O)O"
#target_smiles="c1(cc(O)ccc1)N[C@@H]1N=CNc2n(cnc12)[C@H]1[C@@H]([C@@H]([C@@H](CO[P@](=O)(O)O)O1)O)O"
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
parser.add_argument( "--vector", required=True)
parser.add_argument("--vocab",required=True) 
parser.add_argument("--model")
parser.add_argument( "--save_dir", required=True)
parser.add_argument("--hidden",  type=int,default=450)
parser.add_argument("--seed",  default=None)
parser.add_argument( "--latent", type=int, default=56)
parser.add_argument("--depth", type=int,default=20)
args = parser.parse_args()
print(args)

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

hidden_size = int(args.hidden)
latent_size = int(args.latent)
depth = int(args.depth)
random_seed = int(args.seed)

model = JTNNVAE(vocab, hidden_size, latent_size, 20,3)
model.load_state_dict(torch.load(args.model))
model = model.cuda()

# We load the random seed
np.random.seed(random_seed)
vector_dir=str(args.vector)
# We load the data (y is minued!)
similarity = np.loadtxt('predict/sim1_100.txt')
#cycle_scores = np.loadtxt('cycle_scores.txt')
#logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
#cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)


X = np.loadtxt(vector_dir)
y =-similarity
y = y.reshape((-1, 1))

n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

np.random.seed(random_seed)

iteration = 0
while iteration < 5:
    # We fit the GP
    np.random.seed(iteration * random_seed)
    M = 1
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 100, learning_rate = 0.001)

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print ('Test RMSE: ', error)
    print ('Test ll: ', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print ('Train RMSE: ', error)
    print ('Train ll: ', trainll)

    # We pick the next 60 inputs
    next_inputs = sgp.batched_greedy_ei(60, np.min(X_train, 0), np.max(X_train, 0))
    valid_smiles = []
    new_features = []
    for i in range(60):
        all_vec = next_inputs[i].reshape((1,-1))
        print(all_vec)
        tree_vec,mol_vec = np.hsplit(all_vec, 2)
        tree_vec = create_var(torch.from_numpy(tree_vec).float())
        mol_vec = create_var(torch.from_numpy(mol_vec).float())
        s = model.decode(tree_vec, mol_vec, prob_decode=False)
        if s is not None: 
            valid_smiles.append(s)
            new_features.append(all_vec)
    
    print (len(valid_smiles), "molecules are found")
    valid_smiles = valid_smiles[:50]
    new_features = next_inputs[:50]
    new_features = np.vstack(new_features)
    save_object(valid_smiles, args.save_dir + "/valid_smiles{}.dat".format(iteration))

    import networkx as nx
    from rdkit.Chem import rdmolops

    scores = []
    for i in valid_smiles:
        fps=MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(i))
        score=DataStructs.TanimotoSimilarity(fps,maccs_fp_target)
    #    current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles[ i ]))
#        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles[ i ]))))
#        if len(cycle_list) == 0:
#            cycle_length = 0
#        else:
#            cycle_length = max([ len(j) for j in cycle_list ])
#        if cycle_length <= 6:
#            cycle_length = 0
#        else:
#            cycle_length = cycle_length - 6
#
       # current_cycle_score = -cycle_length
     
    #    current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
      #  current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

   #     score = current_log_P_value_normalized 
        scores.append(-score) #target is always minused
    with open(args.save_dir+"/smiles{}.txt".format(iteration),"w") as fs:
        for i in valid_smiles:
            fs.write(i+"\n")
        
    print (valid_smiles)
    print (scores) 

    save_object(scores, args.save_dir + "/scores{}.dat".format(iteration))

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    iteration += 1
