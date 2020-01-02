import os

import pandas as pd
import numpy as np
import transformers

import pdb

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# [PAD]: 0
# [ANSWER]: 1
# [QUESTION_BODY]: 2

# TODO add CLS??

train_df['text'] = train_df.apply(lambda x: '[CLS] ' + x['question_title'] + ' [SEP] ' +  x['question_body'] +  '[SEP] '  + x['answer'] + '[SEP] ', axis=1)
test_df['text'] = train_df.apply(lambda x:  '[CLS] ' + x['question_title'] + ' [SEP] ' +  x['question_body'] +  '[SEP] '  + x['answer'] + '[SEP] ', axis=1)

df = train_df.append(test_df, sort=False)

df = df.sample(frac=1, random_state=42)

text = df['text'].str.cat(sep='')

with open('data/train_qa_2.txt', 'w+') as f:
    f.write(text)

"""
tokenizer = transformers.BertTokenizer.from_pretrained('model/bert-base-uncased-qa', additional_special_tokens=['[QUESTION_BODY]', '[ANSWER]'])
test = "[QUESTION_BODY] What is your name ? [ANSWER] My name is bob ."
tokens = tokenizer.encode(test, max_length=512, add_special_tokens=True)
print({text: tokens for text, tokens in zip( ['CLS'] + test.split(' ') + ['SEP'], tokens)})
"""