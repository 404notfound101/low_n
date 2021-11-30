#!/usr/bin/env python
# coding: utf-8

# In[33]:


import torch
from transformers import XLNetModel, XLNetTokenizer
import pandas as pd
import numpy as np
import sys
import sklearn
import pickle
import numpy as np
import pandas as pd
import scipy
import random
from torch import nn
from torch.utils import data as data

sys.path.append('../common')
import data_io_utils
import paths
import utils
import constants

import A003_common
import acquisition_policies
import models


# In[28]:


tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)


# In[3]:


model = XLNetModel.from_pretrained("Rostlab/prot_xlnet",mem_len=512)


# In[4]:


training_set_df = pd.read_csv(eval('paths.SARKISYAN_SPLIT_1_FILE'))


# In[20]:


#policy = acquisition_policies.RandomAcquisition()
#training_samples =  policy.acquire_points(training_set_df,256,{})
training_samples = training_set_df["seq"].values


# In[31]:


training_seq = []
for seq in training_samples.tolist():
    new_string = ""
    for i in range(len(seq)-1):
        new_string += seq[i]
        new_string += " "
    new_string += seq[-1]
    training_seq.append(new_string)
training_seq


# In[32]:


ids = tokenizer.batch_encode_plus(training_seq, add_special_tokens=True, padding=True)


# In[34]:


class MyDataSet(data.Dataset):
    def __init__(self, input_ids, attention_mask, qfunc):
        super(MyDataSet, self).__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.qfunc = qfunc
  
    def __len__(self):
        return self.input_ids.shape[0]
  
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.qfunc[idx]
loader = data.DataLoader(MyDataSet(np.array(ids["input_ids"]), np.array(ids["attention_mask"]), training_set_df["quantitative_function"].values), 500, True)


# In[15]:


model.cuda()


# In[16]:


model.eval()


# In[ ]:

embedding = []
qfunc_set = []
count = 0
for seqs in loader:
    count+=1
    input_ids, attention_mask, qfunc = seqs
    input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
    print(input_ids.size())
    with torch.no_grad():
        output = model(input_ids,attention_mask)
    output = output.last_hidden_state * torch.unsqueeze(attention_mask, 1).tile((1,1024,1)).permute(0,2,1)
    print(output.size())
    output = torch.mean(output[:,:-2,:],1)
    print(output.size())
    output = output.cpu().detach().numpy()
    embedding.append(output)
    qfunc_set.append(qfunc.numpy())
    print(count)
embedding = np.concatenate(embedding)
qfunc_set = np.concatenate(qfunc_set)
emb_qfunc = {"embedding":embedding, "qfunc":qfunc_set}
pickle.dump( emb_qfunc, open( "validation_embedding_xlnet.p", "wb" ) )
print("done")

