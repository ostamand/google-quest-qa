import os

import torch
import torch.nn as nn
import transformers

class BertOnQuestions(nn.Module):
    def  __init__(self, output_shape, model_dir, **kwargs):
        super(BertOnQuestions, self).__init__()
        
        self.bert = transformers.BertModel.from_pretrained(model_dir)
        self.fc_dp = nn.Dropout(kwargs['fc_dp'])
        #self.fc = nn.Linear(self.bert.config.hidden_size, output_shape)
        self.fc = nn.Linear(self.bert.config.hidden_size*2, output_shape)
        self.avg_pool = nn.AvgPool1d(512)
        self.max_pool = nn.MaxPool1d(512)
        
        # prepare parameters for optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': kwargs['bert_wd']},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.fc.named_parameters()], 'weight_decay': kwargs['fc_wd']}
        ]
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.fc(self.pooled_dp(pooled))
        """
        seq, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x_avg = self.avg_pool(seq.permute([0,2,1])).squeeze() # bs, 768
        x_max = self.max_pool(seq.permute([0,2,1])).squeeze()
        x = torch.cat([x_avg, x_max], axis=1)
        out = self.fc(self.fc_dp(x))
        return out

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

class BertOnQA(nn.Module):
    def  __init__(self, output_shape, model_dir, **kwargs):
        super(BertOnQA, self).__init__()
        
        self.bert = transformers.BertModel.from_pretrained(model_dir)
        self.fc_dp = nn.Dropout(kwargs['fc_dp'])
        self.fc = nn.Linear(self.bert.config.hidden_size*2, output_shape)
        self.avg_pool = nn.AvgPool1d(512)
        self.max_pool = nn.MaxPool1d(512)

        """
        self.head = nn.Sequential([
            nn.BatchNorm1d(self.bert.config.hidden_size*2),
            nn.Dropout(0.4),
            nn.Linear(self.bert.config.hidden_size*2, 256, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, output_shape)
        ])
        """
        
        # prepare parameters for optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': kwargs['bert_wd']},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.fc.named_parameters()], 'weight_decay': kwargs['fc_wd']}
        ]
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        seq, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # was: out = self.fc(self.pooled_dp(pooled))
        x_avg = self.avg_pool(seq.permute([0,2,1])).squeeze() # bs, 768
        x_max = self.max_pool(seq.permute([0,2,1])).squeeze()
        x = torch.cat([x_avg, x_max], axis=1)
        # TODO add batchnorm, dropout, linear
        out = self.fc(self.fc_dp(x))
        return out
    
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
