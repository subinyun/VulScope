# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


    
class SimilarityModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer):
        super(SimilarityModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
    
        
    def forward(self, input_ids=None,p_input_ids=None,n_input_ids=None,labels=None): 
        bs,_=input_ids.size()
        input_ids=torch.cat((input_ids,p_input_ids,n_input_ids),0)
        
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))
        if len(outputs) > 1:
            outputs = outputs[1]
        else:
            outputs = outputs[0][:, 0, :]
        outputs=outputs.split(bs,0)
        
        prob_1=(outputs[0]*outputs[1]).sum(-1)
        prob_2=(outputs[0]*outputs[2]).sum(-1)
        temp=torch.cat((outputs[0],outputs[1]),0)
        temp_labels=torch.cat((labels,labels),0)
        prob_3= torch.mm(outputs[0],temp.t())
        mask=labels[:,None]==temp_labels[None,:]
        prob_3=prob_3*(1-mask.float())-1e9*mask.float()
        
        prob=torch.softmax(torch.cat((prob_1[:,None],prob_2[:,None],prob_3),-1),-1)
        loss=torch.log(prob[:,0]+1e-10)
        loss=-loss.mean()
        return loss,outputs[0]
