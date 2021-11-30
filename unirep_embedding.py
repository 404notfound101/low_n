#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from torch.utils import data as data
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
checkpoint = torch.load("fp16_64_trial_fine_turning.pt")
model.load_state_dict(checkpoint['model_state_dict'])
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
model.feedforward = Identity()
model.eval()


# In[ ]:


training_set_df = pd.read_csv(eval('paths.SARKISYAN_SPLIT_0_FILE'))
#policy = acquisition_policies.RandomAcquisition()
#training_samples =  policy.acquire_points(training_set_df,2000,{})

class MyDataSet(data.Dataset):
    def __init__(self, seq, qfunc):
        super(MyDataSet, self).__init__()
        self.seq = seq
        self.qfunc = qfunc
  
    def __len__(self):
        return self.seq.shape[0]
  
    def __getitem__(self, idx):
        return self.seq[idx], self.qfunc[idx]

sequence = training_set_df["seq"].values


loader = data.DataLoader(MyDataSet(sequence,  training_set_df["quantitative_function"].values), 1000, True)
# In[ ]:
tokenizer = TAPETokenizer(vocab='unirep')
count = 0
for seqs in loader:
    count+=1
    tokens = []
    for i in range(len(seqs[0])):
        token_ids = torch.tensor([tokenizer.encode(seqs[0][i])])
        tokens.append(token_ids)
    inputs = torch.stack(tokens,dim = 0).view(-1,240)
    print(inputs.size())
    inputs = inputs.cuda()
    model.cuda()
    output = model(inputs)
    output = torch.mean(output[0][:,1:-1,:],1)
    output = output.cpu().detach().numpy()
    qfunc = seqs[1]
    emb_qfunc = {"embedding":output, "qfunc":qfunc}
    pickle.dump( emb_qfunc, open( "1000_training_embedding_fp16_65epoch_" + str(count) + ".p", "wb" ) )
    print(count)
print("done")
