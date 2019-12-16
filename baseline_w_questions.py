from functools import partial
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
import re
import pickle
import os
import gc
import time
from multiprocessing import Process

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_hub as hub
import transformers
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from sentence_encodings import embeddings_from_col, get_dist_features
from transformer_encodings import get_features
from constants import text_columns, targets
from helpers_tf import go_deterministic
from callbacks import SpearmanrCallback
from get_hidden_states_questions import run as calculate_q_hidden_states

import pdb

def get_model(input_size, output_size):
    inputs = tf.keras.layers.Input(shape=(input_size))
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_size, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

class Baseline():

    def __init__(self, params=None):
        self.params = self.default_params() if not params else params
        self.features, self.enc = None, None

    def _get_distilbert_hidden_states(self, model, tokenizer, df):
        answer = get_features(model, tokenizer, self.params['maxlen'], df.answer.values)
        question_title = get_features(model, tokenizer, self.params['maxlen'], df.question_title.values)
        return np.hstack([question_title, answer])

    @property 
    def model_dir(self):
        return self.params['model_dir']

    # calculate_features(data, 0)
    # everything that can be cached is.
    def calculate_features(self, data: Dict[str, pd.DataFrame], fold_n: int):
        # will re-use calculated stuff if called more than once
        save_features = False
        if self.features is None:
            self.features = {k: {} for k,v in data.items()} if self.features is None else self.features
            save_features = True

        K.clear_session()

        # bert question hidden states

        print('Calculating question body hidden states...')
        for k, v in data.items():
            self.features[k]['question_hiddens']  = calculate_q_hidden_states(
                v, 
                self.model_dir, 
                self.model_dir / f"bert_questions_fold_{fold_n}" / 'model_state_dict.pth'
            )
            
        # distilbert hidden states

        if 'distilbert_hidden_states' not in self.features[[*self.features][0]]:
            print('Calculating Distilbert hidden states...')
            bert = transformers.TFDistilBertModel.from_pretrained(self.model_dir / 'distilbert-base-uncased')
            bert.compile(loss='binary_crossentropy')

            tokenizer = transformers.DistilBertTokenizer.from_pretrained(self.model_dir / 'distilbert-base-uncased')

            for k,v in data.items():
                self.features[k]['distilbert_hidden_states'] = self._get_distilbert_hidden_states(bert, tokenizer, v)

            K.clear_session()

        # universal sentence encoder

        if 'sentence_embeds' not in self.features[[*self.features][0]]:
            print('Calculating universal sentence embeddings...')
            embed = hub.load(str(self.model_dir / 'universal-sentence-encoder-large-5'))

            for k,v in data.items():
                embeddings = embeddings_from_col(v, text_columns, embed)
                self.features[k]['sentence_embeds'] = np.hstack([embed for i, embed in embeddings.items()])
                self.features[k]['sentence_embeds_dist'] = get_dist_features(embeddings)

        # category

        if 'category' not in self.features[[*self.features][0]]:
            print('Calculating categorical features...')
            find = re.compile(r"^[^.]*")
            data['train']['netloc'] = data['train']['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
            
            if self.enc is None:
                self.enc = OneHotEncoder(handle_unknown='ignore')
                self.enc.fit(data['train'][['category', 'netloc']].values)
            
            for k,v in data.items():
                data[k]['netloc'] = data['train']['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
                self.features[k]['category'] = self.enc.transform(data[k][['category', 'netloc']].values).toarray()

        if save_features:
            with open(params['temp_dir'] / 'features.pickle', 'wb') as f:
                pickle.dump((self.features, self.enc), f)

    def load_features(self):
        path = params['temp_dir'] / 'features.pickle'
        if not os.path.exists(path):
            return
        with open(params['temp_dir'] / 'features.pickle', 'rb') as f:
            self.features, self.enc = pickle.load(f)

    def predict(self, df, restore_folder):
        pass

    def train_all(self, data, out_dir='baseline_w_questions'):
        # use process. tensorflow takes all the gpu...
        f = partial(self.train, data)
        for i in range(5):
            p = Process(target=f, args=(i, out_dir))
            p.start()
            p.join()

        # extract results
        all_test_preds = []
        rhos = []
        for i in range(5):
            with open(self.model_dir / out_dir / f"results_fold_{i}.pickle", 'rb') as f:
                rho, test_preds = pickle.load(f)
            all_test_preds.append(test_preds)
            rhos.append(rho)
        #None if np.any(all_test_preds is None) else all_test_preds

        # also save encoder in the same folder
        with open(self.model_dir / out_dir / f"results_fold_{i}.pickle", 'wb') as f:
            pickle.dump(self.enc, f)

        return rhos, all_test_preds

    # to speed uo things can do training and inference at the same time 
    def train(self, data: Dict[str, pd.DataFrame], fold_n: int, out_dir='baseline_w_questions'):
        self.calculate_features(data, fold_n)

        x = {}

        for k, v in self.features.items():
            f = self.features[k]
            x[k] = np.hstack([
                f['sentence_embeds'],
                f['category'],
                f['sentence_embeds_dist'],
                f['question_hiddens'],
                f['distilbert_hidden_states']
            ])

        y_train = data['train'][targets].values
        x_train = x['train']
        x_test = x['test'] if 'test' in x else None

        tr_ids = pd.read_csv(os.path.join('data', f"train_ids_fold_{fold_n}.csv"))['ids'].values
        val_ids = pd.read_csv(os.path.join('data', f"valid_ids_fold_{fold_n}.csv"))['ids'].values

        x_tr = x_train[tr_ids]
        x_vl = x_train[val_ids]

        y_tr = y_train[tr_ids]
        y_vl = y_train[val_ids]

        go_deterministic(self.params['seed'])

        model = get_model(x_tr.shape[1], y_tr.shape[1])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-4),
            loss=['binary_crossentropy']
        )

        # TODO change checkpoint name
        cb = SpearmanrCallback((x_vl, y_vl))

        history = model.fit(
            x_tr,
            y_tr,
            validation_data=(x_vl, y_vl),
            epochs=self.params['epochs'],
            batch_size=self.params['bs'],
            verbose=True,
            callbacks=[cb]
        )

        if not os.path.exists(self.model_dir / out_dir):
            os.mkdir(self.model_dir / out_dir)
        model.save_weights(str(self.model_dir / out_dir / f"best_weights_fold_{fold_n}.h5"))

        test_preds = None
        if x_test is not None:
            test_preds = model.predict(x_test)

        result = (cb.best_rho, test_preds)
        with open(self.model_dir / out_dir / f"results_fold_{fold_n}.pickle", 'wb') as f:
            pickle.dump(result, f)

    @classmethod
    def default_params(cls):
        params = {
            'temp_dir': Path('../.tmp'),
            'model_dir': Path('../model'),
            'maxlen': 512, 
            'seed': 42,
            'epochs': 100, 
            'bs': 32
        }
        return params

if __name__ == '__main__':
    params = Baseline.default_params()
    params['model_dir'] = Path('model')
    params['temp_dir'] = Path('.tmp')

    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    data = {'train': train_df, 'test': test_df}

    model = Baseline(params)
    model.load_features()

    rhos, test_preds_train = model.train_all(data, out_dir='test_w_questions') # train all folds

    print(f"rho val: {np.mean(rhos):.4f} +- {np.std(rhos):.4f}")

    # check one fold vs predict function result

    #model = Baseline()
    #test_preds_predict = model.predict(test_df, params['model_dir'] / 'test_w_questions')
