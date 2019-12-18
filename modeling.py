import os

import torch
import torch.nn as nn
import transformers

class BertOnQuestions(nn.Module):
    def  __init__(self, output_shape, model_dir, **kwargs):
        super(BertOnQuestions, self).__init__()
        
        self.bert = transformers.BertModel.from_pretrained(os.path.join(model_dir, 'bert-base-uncased'))
        self.pooled_dp = nn.Dropout(kwargs['fc_dp'])
        self.fc = nn.Linear(self.bert.config.hidden_size, output_shape)
        
        # prepare parameters for optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': kwargs['bert_wd']},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.fc.named_parameters()], 'weight_decay': kwargs['fc_wd']}
        ]
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.fc(self.pooled_dp(pooled))
    
    def train_head_only(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc.requires_grad = True
        
    def train_all(self):
        for param in self.parameters():
            param.requires_grad = True
    
    @classmethod
    def default_params(cls):
        return {
            'fc_dp': 0.4, 
            'bert_wd': 0.01,
            'fc_wd': 0.0
        }