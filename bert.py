#!/usr/bin/env python
# coding: utf-8

# In[33]:


import torch
from transformers import BertModel, BertTokenizer
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




data_file = "soluprot_data/short_test_seqs.fasta" #data path (fasta)
output_file = "solu_bert_test_embedding.p" #output path (pickle)
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")
training_set = []
seq_id = []
for seq_record in SeqIO.parse(data_file, "fasta"):
    seq = str(seq_record.seq)
    new_string = ""
    for i in range(len(seq)-1):
        new_string += seq[i]
        new_string += " "
    new_string += seq[-1]
    seq_id.append(str(seq_record.id))
    training_set.append(new_string)
ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
class MyDataSet(data.Dataset):
    def __init__(self, input_ids, attention_mask):
        super(MyDataSet, self).__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
    def __len__(self):
        return self.input_ids.shape[0]
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
loader = data.DataLoader(MyDataSet(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), 100, False)
model.cuda()
model.eval()
embeddings = []
count = 0
for seq in loader:
    count+=1
    input_ids, input_mask = seq[0],seq[1]
    input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
    with torch.no_grad():
        output = model(input_ids,input_mask)
    output = output[0].cpu().detach().numpy()
    features = [] 
    for seq_num in range(len(output)):
        seq_len = (input_mask[seq_num] == 1).sum()
        seq_emd = output[seq_num][1:seq_len-1]
        features.append(np.mean(seq_emd,axis=0))
    features = np.stack(features)
    print(features.shape)
    embeddings.append(features)
embedding = np.concatenate(embeddings)
print(embedding.shape)
embedding_output = {"embedding":embedding, "seq_ids":seq_id}
pickle.dump( embedding_output, open( output_file, "wb" ) )
print("done")

