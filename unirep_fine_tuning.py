#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from tape import UniRepModel
import time


# In[2]:


from tape import datasets


# In[3]:


from tape import UniRepForLM


# In[4]:


model = UniRepForLM.from_pretrained('babbler-1900')


# In[5]:


from torch import nn


# In[6]:


model.feedforward = nn.Linear(1900, 26)


# In[7]:


embed = datasets.EmbedDataset(data_file="short_training_seqs.fasta",tokenizer="unirep")
embed_validation = datasets.EmbedDataset(data_file="short_test_seqs.fasta",tokenizer="unirep")

# In[8]:


print(embed.__len__())


# In[9]:


from torch.utils import data as data


# In[11]:


loader = data.DataLoader(embed,24,True,collate_fn = embed.collate_fn)
validation_loader = data.DataLoader(embed,24,True,collate_fn = embed.collate_fn)

# In[16]:


import torch.optim as optim
model.cuda()
optimizer = optim.Adam(model.parameters(),lr=0.00001)
model.train()
scaler = torch.cuda.amp.GradScaler()
# In[13]:



# In[21]:

# In[18]:
start_time = time.time()
for epoch in range(131):
    for seq in loader:
        ids, input_ids, input_mask = seq["ids"],seq["input_ids"],seq["input_mask"]
        #if input_ids.size(1) > 280:
            #input_ids, input_mask = input_ids[:,:280], input_mask[:,:280]
        input_ids, input_mask = input_ids.cuda(),input_mask.cuda()
        loss = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input_ids,input_mask)
            imloss = loss(output[0][:,:-1].reshape(-1,26),input_ids[:,1:].reshape(-1))
        scaler.scale(imloss).backward()
        #imloss.backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()
    if epoch%10 == 0:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': imloss}, "solu_epoch"+str(epoch)+"_trial_training.pt")
        print("============= epoch %s time elapsed : %s seconds =============" % (epoch, time.time() - start_time, ))
        model.eval()
        validation_loss_set = []
        for valiseq in validation_loader:
            ids, input_ids, input_mask = valiseq["ids"],valiseq["input_ids"],valiseq["input_mask"]
            input_ids, input_mask = input_ids.cuda(),input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            validation_loss = loss(output[0][:,:-1].reshape(-1,26),input_ids[:,1:].reshape(-1))
            validation_loss_set.append(validation_loss.view(-1,1))
        validation_loss = torch.mean(torch.cat(validation_loss_set))
        model.train()
        print("epoch: {} |Loss: {} |Validation: {}".format(epoch,imloss,validation_loss))
print("---{} seconds---".format(time.time()-start_time))
'''start_time = time.time()
for epoch in range(130):
    for seq in loader:
        ids, input_ids, input_mask = seq["ids"],seq["input_ids"],seq["input_mask"]
        #if input_ids.size(1) > 280:
            #input_ids, input_mask = input_ids[:,:280], input_mask[:,:280]
        input_ids, input_mask = input_ids.cuda(),input_mask.cuda()
        loss = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input_ids,input_mask)
            imloss = loss(output[0][:,:-1].reshape(-1,26),input_ids[:,1:].reshape(-1))
        scaler.scale(imloss).backward()
        #imloss.backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()
    if epoch%10 == 0:
        print("epoch: {} |Loss: {}".format(epoch,imloss))
    if epoch in [10,20,50,64,80,100,129]:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': imloss}, "fp16_"+str(epoch)+"_trial_fine_turning.pt")        
print("---{} seconds---".format(time.time()-start_time)'''