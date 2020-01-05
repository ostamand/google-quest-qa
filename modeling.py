import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

import pdb

class BertOnQuestions(nn.Module):
    def  __init__(self, output_shape, model_dir, **kwargs):
        super(BertOnQuestions, self).__init__()

        config = transformers.BertConfig.from_pretrained(model_dir)
        config.output_hidden_states=True 
        self.bert = transformers.BertModel.from_pretrained(model_dir, config=config)

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

        seq, pooled, hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        """
        x_avg = self.avg_pool(seq.permute([0,2,1]))[:,:,0] # bs, 768
        x_max = self.max_pool(seq.permute([0,2,1]))[:,:,0]
        x = torch.cat([x_avg, x_max], axis=1)
        """

        h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h11 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        #h10 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        #h9  = hidden_states[-4][:, 0].reshape((-1, 1, 768))
        #all_h = torch.cat([h9, h10, h11, h12], axis=1)
        all_h = torch.cat([h11, h12], axis=1)

        mean_pool = torch.mean(all_h, axis=1)
        max_pool, _ = torch.max(all_h, axis=1)

        x = torch.cat([mean_pool, max_pool], axis=1)

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
        self.fc_dp = nn.Dropout(kwargs['fc_dp'] if 'fc_dp' in kwargs else 0.)
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
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': kwargs['bert_wd'] if 'bert_wd' in kwargs else 0.},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.fc.named_parameters()], 'weight_decay': kwargs['fc_wd'] if 'fc_wd' in kwargs else 0.}
        ]
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        seq, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # was: out = self.fc(self.pooled_dp(pooled))
        x_avg = self.avg_pool(seq.permute([0,2,1]))[:,:,0] # bs, 768
        x_max = self.max_pool(seq.permute([0,2,1]))[:,:,0]
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

class CustomBertPooling(nn.Module):
    def __init__(self):
        super(CustomBertPooling, self).__init__()

        # TODO try ReLU & Leaky ReLU

        self.layer_1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh()
            #nn.LeakyReLU(0., inplace=True)
        )

        self.avg_pool = nn.AvgPool1d(512)
        self.max_pool = nn.MaxPool1d(512)

        self.layer_2 = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh()
            #nn.LeakyReLU(0., inplace=True)
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh()
            #nn.LeakyReLU(0., inplace=True)
        )

        self.output_shape = 768*3


    def forward(self, hidden_states):
        # cls hidden states
        h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h11 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h10 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h9  = hidden_states[-4][:, 0].reshape((-1, 1, 768))

        # h12 cls 
        x1 = self.layer_1(h12[:,0,:])

        # h12 seq pooling
        x_avg = self.avg_pool(hidden_states[-1].permute([0,2,1]))[:,:,0]
        x_max = self.max_pool(hidden_states[-1].permute([0,2,1]))[:,:,0]
        x2 = torch.cat([x_avg, x_max], axis=1)
        x2 = self.layer_2(x2)

        # h12, h11, h10, h9 cls pooling
        all_h = torch.cat([h9, h10, h11, h12], axis=1)
        mean_pool = torch.mean(all_h, axis=1)
        max_pool, _ = torch.max(all_h, axis=1)
        x3 = torch.cat([mean_pool, max_pool], axis=1)
        x3 = self.layer_3(x3)

        out = torch.cat([x1, x2, x3], axis=1)

        return out

class BertOnQA_2(nn.Module):
    def  __init__(self, output_shape, model_dir, **kwargs):
        super(BertOnQA_2, self).__init__()

        config = transformers.BertConfig.from_pretrained(model_dir)
        config.output_hidden_states=True 
        self.bert = transformers.BertModel.from_pretrained(model_dir, config=config)
        self.pooling = CustomBertPooling()

        self.layer_1 = torch.nn.Sequential(
           # nn.Dropout(p=0., inplace=True),
            nn.Linear(768*3, output_shape)
        )
        
        # prepare parameters for optimizer
        no_decay = ['bias', 'LayerNorm.weight']

        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': kwargs['bert_wd'] if 'bert_wd' in kwargs else 0.},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.layer_1.named_parameters()], 'weight_decay': 0.},
            {'params': [p for n, p in self.pooling.named_parameters()], 'weight_decay': 0.}
        ]
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        seq, pooled, hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.pooling(hidden_states)
        x = self.layer_1(x)
        return x
    
    def train_head_only(self):
        for param in self.bert.parameters():
            param.requires_grad = False

        for param in self.pooling.parameters():
            param.requires_grad = True

        for param in self.layer_1.parameters():
            param.requires_grad = True
        
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
