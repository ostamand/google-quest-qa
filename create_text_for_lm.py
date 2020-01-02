import os

import pandas as pd
import numpy as np
import transformers
from sklearn.model_selection import train_test_split

import pdb

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# [PAD]: 0
# [ANSWER]: 1
# [QUESTION_BODY]: 2

# TODO add CLS??

train_df['text'] = train_df.apply(lambda x: x['question_title'] + ' ' +  x['question_body'] +  ' '  + x['answer'] + ' ', axis=1)
test_df['text'] = train_df.apply(lambda x:  x['question_title'] + ' ' +  x['question_body'] +  ' '  + x['answer'] + ' ', axis=1)
df = train_df.append(test_df, sort=False)
df = df.sample(frac=1, random_state=42)

train_df, valid_df = train_test_split(df, random_state=42, test_size=0.1)

train_text = train_df['text'].str.cat(sep='')
valid_text = valid_df['text'].str.cat(sep='')

with open('data/train_lm.txt', 'w+') as f:
    f.write(train_text)

with open('data/valid_lm.txt', 'w+') as f:
    f.write(valid_text)