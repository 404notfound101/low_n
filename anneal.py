#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import torch
import tape
import sklearn
import pickle
import numpy as np
import pandas as pd
import scipy
import random
from sklearn.linear_model import LassoLars, LassoLarsCV, Ridge, RidgeCV, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from torch import nn
from tape import TAPETokenizer
from tape import UniRepForLM

sys.path.append('../common')
import data_io_utils
import paths
import utils
import constants

import A003_common
import acquisition_policies
import models


# In[ ]:


model = UniRepForLM.from_pretrained("babbler-1900")
model.feedforward = nn.Linear(1900,26)


# In[ ]:


checkpoint = torch.load("/nfs/homes/wujingh4/Desktop/UniRep-analysis/18000_trial_training.pt")
model.load_state_dict(checkpoint['model_state_dict'])


# In[ ]:


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
model.feedforward = Identity()


# In[ ]:


model.eval()


# In[ ]:


emb_qfunc = pickle.load( open( "save.p", "rb" ) )
embedding = emb_qfunc["embedding"]
qfunc = emb_qfunc["qfunc"]


# In[ ]:


TOP_MODEL_DO_SPARSE_REFIT = True
EnsembledRidge = A003_common.train_ensembled_ridge(
                embedding, 
                qfunc, 
                do_sparse_refit=TOP_MODEL_DO_SPARSE_REFIT, 
                n_members=100, 
                subspace_proportion=0.5, 
                pval_cutoff=0.01, 
                normalize=True
            )


# In[ ]:


SIM_ANNEAL_K = 1
SIM_ANNEAL_INIT_SEQ_MUT_RADIUS = 3
n_chains = 1000
T_max = 0.01*np.ones(n_chains)
GFP_LIB_REGION = [29, 110]
seed = 0
temp_decay_rate = 1.0
sa_n_iter = 3000
nmut_threshold = 7
np.random.seed(seed)
random.seed(seed)


# In[ ]:


def acceptance_prob(f_proposal, f_current, k, T):
    ap = np.exp((f_proposal - f_current)/(k*T))
    ap[ap > 1] = 1
    return ap

def make_n_random_edits(seq, nedits, alphabet=constants.AA_ALPHABET_STANDARD_ORDER,
        min_pos=None, max_pos=None): ## Test
    """
    min_pos is inclusive. max_pos is exclusive
    """
    
    lseq = list(seq)
    lalphabet = list(alphabet)
    
    if min_pos is None:
        min_pos = 0
    
    if max_pos is None:
        max_pos = len(seq)
    
    # Create non-redundant list of positions to mutate.
    l = list(range(min_pos, max_pos))
    nedits = min(len(l), nedits)
    random.shuffle(l)
    pos_to_mutate = l[:nedits]    
    
    for i in range(nedits):
        pos = pos_to_mutate[i]     
        aa_to_choose_from = list(set(lalphabet) - set([seq[pos]]))
                        
        lseq[pos] = aa_to_choose_from[np.random.randint(len(aa_to_choose_from))]
        
    return "".join(lseq)

def propose_seqs(seqs, mu_muts_per_seq, min_pos=None, max_pos=None):
    
    mseqs = []
    for i,s in enumerate(seqs):
        n_edits = np.random.poisson(mu_muts_per_seq[i]-1) + 1
        mseqs.append(make_n_random_edits(s, n_edits, min_pos=min_pos, max_pos=max_pos)) 
        
    return mseqs

def anneal(
        init_seqs, 
        k, 
        T_max, 
        mu_muts_per_seq,
        get_fitness_fn,
        n_iter=1000, 
        decay_rate=0.99,
        min_mut_pos=None,
        max_mut_pos=None):
    
    print('Initializing')
    state_seqs = copy.deepcopy(init_seqs)
    state_fitness, state_fitness_std, state_fitness_mem_pred = get_fitness_fn(state_seqs)
    
    seq_history = [copy.deepcopy(state_seqs)]
    fitness_history = [copy.deepcopy(state_fitness)]
    fitness_std_history = [copy.deepcopy(state_fitness_std)]
    fitness_mem_pred_history = [copy.deepcopy(state_fitness_mem_pred)]
    for i in range(n_iter):
        if i%100 ==0:
            print('Iteration:', i)        
            print('\tProposing sequences.')
        proposal_seqs = propose_seqs(state_seqs, mu_muts_per_seq, 
                min_pos=min_mut_pos, max_pos=max_mut_pos)
        if i%100 == 0:
            print('\tCalculating predicted fitness.')
        proposal_fitness, proposal_fitness_std, proposal_fitness_mem_pred = get_fitness_fn(proposal_seqs)
        
        if i%100 == 0:
            print('\tMaking acceptance/rejection decisions.')
        aprob = acceptance_prob(proposal_fitness, state_fitness, k, T_max*(decay_rate**i))
        
        # Make sequence acceptance/rejection decisions
        for j, ap in enumerate(aprob):
            if np.random.rand() < ap:
                # accept
                state_seqs[j] = copy.deepcopy(proposal_seqs[j])
                state_fitness[j] = copy.deepcopy(proposal_fitness[j])
                state_fitness_std[j] = copy.deepcopy(proposal_fitness_std[j])
                state_fitness_mem_pred[j] = copy.deepcopy(proposal_fitness_mem_pred[j])
            # else do nothing (reject)
            
        seq_history.append(copy.deepcopy(state_seqs))
        fitness_history.append(copy.deepcopy(state_fitness))
        fitness_std_history.append(copy.deepcopy(state_fitness_std))
        fitness_mem_pred_history.append(copy.deepcopy(state_fitness_mem_pred))
        
    return {
        'seq_history': seq_history,
        'fitness_history': fitness_history,
        'fitness_std_history': fitness_std_history,
        'fitness_mem_pred_history': fitness_mem_pred_history,
        'init_seqs': init_seqs,
        'T_max': T_max,
        'mu_muts_per_seq': mu_muts_per_seq,
        'k': k,
        'n_iter': n_iter,
        'decay_rate': decay_rate,
        'min_mut_pos': min_mut_pos,
        'max_mut_pos': max_mut_pos,
    }


# In[ ]:
init_seqs = propose_seqs(
        [constants.AVGFP_AA_SEQ]*(n_chains*3), 
        [SIM_ANNEAL_INIT_SEQ_MUT_RADIUS]*(n_chains*3), 
        min_pos=GFP_LIB_REGION[0], 
        max_pos=GFP_LIB_REGION[1])

init_seqs_0 = init_seqs[:n_chains]
init_seqs_1 = init_seqs[n_chains:n_chains*2]
init_seqs_2 = init_seqs[n_chains*2:n_chains*3]
mu_muts_per_seq = 1.5*np.random.rand(n_chains*3) + 1
mu_muts_per_seq_0 = mu_muts_per_seq[:1000]
mu_muts_per_seq_1 = mu_muts_per_seq[1000:2000]
mu_muts_per_seq_2 = mu_muts_per_seq[2000:3000]
print('mu_muts_per_seq:', mu_muts_per_seq)


# In[ ]:


model.cuda()


# In[ ]:


def get_fitness(seqs):
    tokens = []
    for seq in seqs:
        token_ids = torch.tensor([tokenizer.encode(seq)])
        tokens.append(token_ids)
    inputs = torch.stack(tokens,dim = 0).view(-1,240)
    inputs = inputs.cuda()
    output = model(inputs)
    embedding = torch.mean(output[0][:,1:-1,:],1)
    reps = embedding.cpu().detach().numpy()

    yhat, yhat_std, yhat_mem = EnsembledRidge.predict(reps, 
            return_std=True, return_member_predictions=True)
                
    nmut = utils.levenshtein_distance_matrix(
            [constants.AVGFP_AA_SEQ], list(seqs)).reshape(-1)
        
    mask = nmut > nmut_threshold
    yhat[mask] = -np.inf 
    yhat_std[mask] = 0 
    yhat_mem[mask,:] = -np.inf 
        
    return yhat, yhat_std, yhat_mem 


# In[ ]:
import time
start_time = time.time()

import copy
tokenizer = TAPETokenizer(vocab='unirep')
sa_results_0 = anneal(init_seqs_0,k=SIM_ANNEAL_K,T_max=T_max,mu_muts_per_seq=mu_muts_per_seq_0,get_fitness_fn=get_fitness,n_iter=sa_n_iter,decay_rate=temp_decay_rate,min_mut_pos=GFP_LIB_REGION[0],max_mut_pos=GFP_LIB_REGION[1])
sa_results_1 = anneal(init_seqs_1,k=SIM_ANNEAL_K,T_max=T_max,mu_muts_per_seq=mu_muts_per_seq_1,get_fitness_fn=get_fitness,n_iter=sa_n_iter,decay_rate=temp_decay_rate,min_mut_pos=GFP_LIB_REGION[0],max_mut_pos=GFP_LIB_REGION[1])
sa_results_2 = anneal(init_seqs_2,k=SIM_ANNEAL_K,T_max=T_max,mu_muts_per_seq=mu_muts_per_seq_2,get_fitness_fn=get_fitness,n_iter=sa_n_iter,decay_rate=temp_decay_rate,min_mut_pos=GFP_LIB_REGION[0],max_mut_pos=GFP_LIB_REGION[1])

pickle.dump( sa_results_0, open( "result24_trial6_seed0_3000_7_0.p", "wb" ) )
pickle.dump( sa_results_1, open( "result24_trial6_seed0_3000_7_1.p", "wb" ) )
pickle.dump( sa_results_2, open( "result24_trial6_seed0_3000_7_2.p", "wb" ) )



print("--- %s seconds ---" % (time.time() - start_time))
