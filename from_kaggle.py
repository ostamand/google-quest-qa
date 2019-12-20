#pip install bert-for-tf2
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import gc
import os
from scipy.stats import spearmanr
from math import floor, ceil
from scipy.stats import rankdata
import wandb
from wandb.keras import WandbCallback

from helpers_tf import go_deterministic
from callbacks import LROneCycle, SpearmanrCallback

import pdb

np.set_printoptions(suppress=True)

def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(tokenizer, title, question, answer, max_sequence_length, 
                t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
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
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        t, q, a = _trim_input(tokenizer, t, q, a, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])

def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)

class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, test_data, batch_size=16, out_dir = '.tmp', fold=None):
        self.out_dir = out_dir
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        
        self.batch_size = batch_size
        self.fold = fold
        
    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(
                self.valid_inputs, 
                batch_size=self.batch_size
            )
        )
        
        rho_val = compute_spearmanr(
            self.valid_outputs, 
            np.average(self.valid_predictions, axis=0)
        )
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        #TODO only if better
        #TODO log to wandb
        if self.fold is not None:
            self.model.save_weights(f'{self.out_dir}/bert-base-{self.fold}-{epoch}.h5py')
        
        if self.test_inputs is not None:
            self.test_predictions.append(
                self.model.predict(self.test_inputs, batch_size=self.batch_size)
            )

def bert_model(model_path, maxlen):
    input_word_ids = tf.keras.layers.Input(
        (maxlen,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (maxlen,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (maxlen,), dtype=tf.int32, name='input_segments')
    
    bert_layer = hub.KerasLayer(model_path, trainable=True)
    
    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    #x_max = tf.keras.layers.GlobalMaxPooling1D()(sequence_output)
    #x = tf.keras.layers.Concatenate()([x, x_max])
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], 
        outputs=out
    )
    
    return model    

def main(**args):
    if not os.path.exists(args['out_dir']):
        os.mkdir(args['out_dir'])

    tokenizer = tokenization.FullTokenizer(os.path.join(args['model_dir'], 'assets/vocab.txt'), True)
    
    df_train = pd.read_csv(os.path.join(args['data_dir'], 'train.csv'))
    df_test = pd.read_csv(os.path.join(args['data_dir'], 'test.csv'))
    df_sub = pd.read_csv(os.path.join(args['data_dir'], 'sample_submission.csv'))

    output_categories = list(df_train.columns[11:])
    input_categories = list(df_train.columns[[1,2,5]])

    outputs = compute_output_arrays(df_train, output_categories)
    inputs = compute_input_arays(df_train, input_categories, tokenizer, args['maxlen'])
    test_inputs = compute_input_arays(df_test, input_categories, tokenizer, args['maxlen'])

    if args['fold'] is not None:
        tr_ids =  pd.read_csv(os.path.join(args['data_dir'],  f"train_ids_fold_{args['fold']}.csv"))['ids'].values
        val_ids = pd.read_csv(os.path.join(args['data_dir'],  f"valid_ids_fold_{args['fold']}.csv"))['ids'].values

    K.clear_session()
    go_deterministic(args['seed'])

    model = bert_model(args['model_dir'])

    tags = [] if args['fold'] is None else [args['fold']]
    tags.append('from_kaggle')

    wandb.init(project='google-quest-qa', tags=tags)

    train_inputs = [inputs[i][tr_ids] for i in range(3)]
    train_outputs = outputs[tr_ids]

    valid_inputs = [inputs[i][val_ids] for i in range(3)]
    valid_outputs = outputs[val_ids]

    cb = SpearmanrCallback((valid_inputs, valid_outputs), restore=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    cycle = LROneCycle(train_inputs[0].shape[0])

    callbacks = [
        cycle, 
        cb, 
        WandbCallback()
    ]

    history = model.fit(
        train_inputs, 
        train_outputs, 
        epochs=args['epochs'], 
        batch_size=args['bs'], 
        callbacks=callbacks
    )

# python3 from_kaggle.py --fold 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--out_dir", default="outputs/baseline_on_all", type=str)
    parser.add_argument("--model_dir", default="model/bert_en_uncased_L-12_H-768_A-12", type=str)
    parser.add_argument("--fold", default=None, type=int)
    parser.add_argument("--maxlen", default=512, type=int)
    parser.add_argument("--bs", default=8, type=int)

    args = parser.parse_args()

    main(**args.__dict__)