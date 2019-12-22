import math
import os

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class DatasetQA(Dataset):

    # TODO variable maxlen. for now fixed at 512
    def __init__(self, df, tokenizer, ids=None, max_len_q_b=150, max_len_q_t=30):
        super(DatasetQA, self).__init__()

        #df = df.iloc[:10] # for dev
        
        df['q_b_tokens'] = df['question_body'].apply(lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=False))
        df['q_t_tokens'] = df['question_title'].apply(lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=False))
        df['a_tokens'] = df['answer'].apply(lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=False))
        
        # [PAD]: 0
        # [ANSWER]: 1
        # [QUESTION_BODY]: 2
        def process(row, how=0):
            # token ids:    [CLS] [QUESTION_BODY] question body [ANSWER] answer [SEP] [PAD]
            # token types:  0    ...                            1 ...
            # TODO token types = 0 for the [PAD]?
            if how == 0:
                tokens = [tokenizer.cls_token_id] + [2] + (512-2)*[tokenizer.pad_token_id] 
            
                len_q = np.min([max_len_q_b, len(row['q_b_tokens'])])

                if len(row['a_tokens']) >= 512-4-len_q:
                    # need to truncate the answer and possibly the question
                    question_trunc = row['q_b_tokens'][:len_q]
                    answer_trunc = row['a_tokens'][:512-4-len_q]
                else: 
                    # full answer and maximum question length
                    answer_trunc = row['a_tokens']
                    question_trunc = row['q_b_tokens'][:512-4-len(answer_trunc)]
            
                combined = question_trunc + [1] + answer_trunc + [tokenizer.sep_token_id]
                tokens[2:2+len(combined)] = combined

                len_q += 2 # to consider special tokens
                token_types = [0] * len_q + (512-len_q) * [1]

                return tokens, token_types

            # token ids:    [CLS] question body [SEP] answer [SEP] [PAD]
            # token types:  0    ...             0    1 ...        0 
            if how == 1:
                tokens = [tokenizer.cls_token_id] + (512-1)*[tokenizer.pad_token_id] 

                len_q = np.min([max_len_q_b, len(row['q_b_tokens'])])

                if len(row['a_tokens']) >= 512-3-len_q:
                    # need to truncate the answer and possibly the question
                    question_trunc = row['q_b_tokens'][:len_q]
                    answer_trunc = row['a_tokens'][:512-3-len_q]
                else: 
                    # full answer and maximum question length
                    answer_trunc = row['a_tokens']
                    question_trunc = row['q_b_tokens'][:512-3-len(answer_trunc)]
                
                combined = question_trunc + [tokenizer.sep_token_id] + answer_trunc + [tokenizer.sep_token_id]
                tokens[1:1+len(combined)] = combined

                token_types = [0] * (len(question_trunc)+2) + (len(answer_trunc)+1) * [1] + (512 - len(answer_trunc) - len(question_trunc) - 3) * [0]

                return tokens, token_types

            # token ids:    [CLS] question title [SEP] question body [SEP] answer [SEP] [PAD]
            # token types:  0     0 ...          0     1 ...         1     1 ...  1     0 ...
            if how==2:
                tokens = [tokenizer.cls_token_id] + (512-1)*[tokenizer.pad_token_id] 

                len_q_b = np.min([max_len_q_b, len(row['q_b_tokens'])])
                len_q_t = np.min([max_len_q_t, len(row['q_t_tokens'])])

                len_q = len_q_b + len_q_t

                if len(row['a_tokens']) >= 512-4-len_q:
                    # need to truncate the answer and possibly the questions
                    question_title_trunc = row['q_t_tokens'][:len_q_t]
                    question_body_trunc =  row['q_b_tokens'][:len_q_b]
                    answer_trunc = row['a_tokens'][:512-4-len_q_t-len_q_b]
                else:
                    # full answer and maximum question length
                    answer_trunc = row['a_tokens']
                    
                    # try with full title and truncated question body
                    if len(answer_trunc) + len(row['q_b_tokens']) + len_q_t >= 512-4:
                        
                        question_title_trunc = row['q_t_tokens']
                        question_body_trunc =  row['q_b_tokens'][:512-4-len(answer_trunc)-len(question_title_trunc)]

                        if len(question_title_trunc) + len(question_body_trunc) + len(answer_trunc) > 512-4:
                            # need to truncate title also for now will truncate both
                            # TODO does not happen in the train dataset
                            question_title_trunc = row['q_t_tokens'][:len_q_t]
                            question_body_trunc =  row['q_b_tokens'][:len_q_b]

                    # full question body, question title and answer
                    else:
                        question_body_trunc = row['q_b_tokens']
                        question_title_trunc = row['q_t_tokens']

                combined = question_title_trunc + [tokenizer.sep_token_id] + question_body_trunc + [tokenizer.sep_token_id] + answer_trunc + [tokenizer.sep_token_id]
                tokens[1:1+len(combined)] = combined

                # TODO change token_types. 0 for question, 1 for answer
                #token_types = [0] * (len(question_title_trunc)+2) + (len(question_body_trunc)+len(answer_trunc)+2) * [1] + (512 - len(answer_trunc) - len(question_body_trunc) - len(question_title_trunc)  - 4) * [0]
                token_types = [0] * (len(question_title_trunc)+len(question_body_trunc)+3) + (len(answer_trunc)+1) * [1] + (512 - len(answer_trunc) - len(question_body_trunc) - len(question_title_trunc)  - 4) * [0]

                return tokens, token_types

            if how == 3:
                max_sequence_length=512
                t_max_len=30
                q_max_len=239
                a_max_len=239

                t = tokenizer.tokenize(row['question_title'])
                q = tokenizer.tokenize(row['question_body'])
                a = tokenizer.tokenize(row['answer'])
    
                t_len = len(t)
                q_len = len(q)
                a_len = len(a)

                if (t_len+q_len+a_len+4) > max_sequence_length:
                    
                    if t_max_len > t_len:
                        t_new_len = t_len
                        a_max_len = a_max_len + math.floor((t_max_len - t_len)/2)
                        q_max_len = q_max_len + math.ceil((t_max_len - t_len)/2)
                    else:
                        t_new_len = t_max_len
                
                    if a_max_len > a_len:
                        a_new_len = a_len 
                        q_new_len = q_max_len + (a_max_len - a_len)
                    elif q_max_len > q_len:
                        a_new_len = a_max_len + (q_max_len - q_len)
                        q_new_len = q_len
                    else:
                        a_new_len = a_max_len
                        q_new_len = q_max_len
                        
                    if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
                        raise ValueError("New sequence length should be %d, but is %d" 
                                        % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
                    
                    t = t[:t_new_len]
                    q = q[:q_new_len]
                    a = a[:a_new_len]

                stoken = ["[CLS]"] + t + ["[SEP]"] + q + ["[SEP]"] + a + ["[SEP]"]
                tokens = tokenizer.convert_tokens_to_ids(stoken)
                tokens = tokens + [0] * (max_sequence_length-len(tokens))

                # token types

                if len(tokens)>max_sequence_length:
                    raise IndexError("Token length more than max seq length!")

                segments = []
                first_sep = True
                current_segment_id = 0
                for token in stoken:
                    segments.append(current_segment_id)
                    if token == "[SEP]":
                        if first_sep:
                            first_sep = False 
                        else:
                            current_segment_id = 1
                token_types = segments + [0] * (max_sequence_length - len(stoken))

                return tokens, token_types

        df['all'] = df.apply(lambda x: process(x, how=2), axis=1)

        if targets[0] in df.columns:
            self.labels = df[targets].values.astype(np.float32)
        else: 
            self.labels = None

        self.tokens = np.stack(df['all'].apply(lambda x: x[0]).values).astype(np.long)
        self.token_types = np.stack(df['all'].apply(lambda x: x[1]).values).astype(np.long)

        if ids is not None:
            if self.labels is not None:
                self.labels = self.labels[ids]

            self.tokens = self.tokens[ids]
            self.token_types = self.token_types[ids]

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        labels = self.labels[idx] if self.labels is not None else []
        return self.tokens[idx], self.token_types[idx], labels
        