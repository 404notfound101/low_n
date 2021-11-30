#!/usr/bin/env python
# coding: utf-8

# In[33]:


import torch
from transformers import ElectraTokenizer, ElectraForPreTraining, ElectraForMaskedLM, ElectraModel
import pandas as pd
import numpy as np
import sys
import os
import requests
from tqdm.auto import tqdm
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
generatorModelUrl = 'https://www.dropbox.com/s/5x5et5q84y3r01m/pytorch_model.bin?dl=1'
discriminatorModelUrl = 'https://www.dropbox.com/s/9ptrgtc8ranf0pa/pytorch_model.bin?dl=1'

generatorConfigUrl = 'https://www.dropbox.com/s/9059fvix18i6why/config.json?dl=1'
discriminatorConfigUrl = 'https://www.dropbox.com/s/jq568evzexyla0p/config.json?dl=1'

vocabUrl = 'https://www.dropbox.com/s/wck3w1q15bc53s0/vocab.txt?dl=1'
downloadFolderPath = 'models/electra/'
discriminatorFolderPath = os.path.join(downloadFolderPath, 'discriminator')
generatorFolderPath = os.path.join(downloadFolderPath, 'generator')

discriminatorModelFilePath = os.path.join(discriminatorFolderPath, 'pytorch_model.bin')
generatorModelFilePath = os.path.join(generatorFolderPath, 'pytorch_model.bin')

discriminatorConfigFilePath = os.path.join(discriminatorFolderPath, 'config.json')
generatorConfigFilePath = os.path.join(generatorFolderPath, 'config.json')

vocabFilePath = os.path.join(downloadFolderPath, 'vocab.txt')
if not os.path.exists(discriminatorFolderPath):
    os.makedirs(discriminatorFolderPath)
if not os.path.exists(generatorFolderPath):
    os.makedirs(generatorFolderPath)
def download_file(url, filename):
    response = requests.get(url, stream=True)
    with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                    total=int(response.headers.get('content-length', 0)),
                    desc=filename) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)
if not os.path.exists(generatorModelFilePath):
    download_file(generatorModelUrl, generatorModelFilePath)

if not os.path.exists(discriminatorModelFilePath):
    download_file(discriminatorModelUrl, discriminatorModelFilePath)
    
if not os.path.exists(generatorConfigFilePath):
    download_file(generatorConfigUrl, generatorConfigFilePath)

if not os.path.exists(discriminatorConfigFilePath):
    download_file(discriminatorConfigUrl, discriminatorConfigFilePath)
    
if not os.path.exists(vocabFilePath):
    download_file(vocabUrl, vocabFilePath)
            
tokenizer = ElectraTokenizer(vocabFilePath, do_lower_case=False )


# In[3]:


model = ElectraModel.from_pretrained(discriminatorFolderPath)


# In[4]:


training_set_df = pd.read_csv(eval('paths.SARKISYAN_SPLIT_0_FILE'))


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
loader = data.DataLoader(MyDataSet(np.array(ids["input_ids"]), np.array(ids["attention_mask"]), training_set_df["quantitative_function"].values), 1000, True)


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
    print(output[0].size())
    output = torch.mean(output[0][:,1:-1,:],1)
    output = output.cpu().detach().numpy()
    embedding.append(output)
    qfunc_set.append(qfunc.numpy())
    print(count)
embedding = np.concatenate(embedding)
qfunc_set = np.concatenate(qfunc_set)
emb_qfunc = {"embedding":embedding, "qfunc":qfunc_set}
pickle.dump( emb_qfunc, open( "training_embedding_electra.p", "wb" ) )
print("done")

