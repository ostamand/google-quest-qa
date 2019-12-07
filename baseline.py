from functools import partial
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
import re
import pickle

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
from helpers import go_deterministic
from callbacks import SpearmanrCallback

import pdb

def get_model(input_size, output_size):
    inputs = tf.keras.layers.Input(shape=(input_size))
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_size, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

class Baseline():

    def __init__(self, params, data=None):
        self.params = params
        self.features = None
        self.data = data  

    def _get_bert_hidden_states(self, model, tokenizer, df):
        question_body = get_features(model, tokenizer, self.params['maxlen'], df.question_body.values)
        answer = get_features(model, tokenizer, self.params['maxlen'], df.answer.values)
        question_title = get_features(model, tokenizer, self.params['maxlen'], df.question_title.values)
        return np.hstack([question_body, question_title, answer])

    def load_features(self, path=None):
        path = path if path else self.params['temp_dir'] / 'features.pickle'
        with open(path, 'rb') as f:
            self.features = pickle.load(f)

    def calculate_features(self, data: Dict[str, pd.DataFrame], save=False):
        self.data = data
        model_dir = self.params['model_dir']

        # distilbert hidden states

        K.clear_session()

        bert = transformers.TFDistilBertModel.from_pretrained(model_dir / 'distilbert-base-uncased')
        bert.compile(loss='binary_crossentropy')

        tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_dir / 'distilbert-base-uncased')

        self.features = {k: {} for k,v in data.items()}
        for k,v in data.items():
            self.features[k]['distilbert_hidden_states'] = self._get_bert_hidden_states(bert, tokenizer, v)

        # universal sentence encoder

        K.clear_session()

        embed = hub.load(str(model_dir / 'universal-sentence-encoder-large-5'))

        for k,v in data.items():
            embeddings = embeddings_from_col(v, text_columns, embed)
            self.features[k]['sentence_embeds'] = np.hstack([embed for i, embed in embeddings.items()])
            self.features[k]['sentence_embeds_dist'] = get_dist_features(embeddings)

        # category

        enc = OneHotEncoder(handle_unknown='ignore')

        find = re.compile(r"^[^.]*")
        data['train']['netloc'] = data['train']['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

        enc.fit(data['train'][['category', 'netloc']].values)

        for k,v in data.items():
            data[k]['netloc'] = data['train']['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
            self.features[k]['category'] = enc.transform(data[k][['category', 'netloc']].values).toarray()

        # save
        if save:
            with open(self.params['temp_dir'] / 'features.pickle', 'wb') as f:
                pickle.dump(self.features, f)

    def train(self):
        x = {}

        for k,v in self.features.items():
            f = self.features[k]
            x[k] = np.hstack([
                f['sentence_embeds'], 
                f['category'],
                f['sentence_embeds_dist'], 
                f['distilbert_hidden_states']
            ])
        
        y_train = self.data['train'][targets].values
        x_train = x['train']
        x_test = x['test'] if 'test' in x else None

        rhos = []
        test_preds = []
        models = []

        kf = KFold(n_splits=self.params['n_splits'], random_state=self.params['seed'], shuffle=True)

        for i, (tr, val) in enumerate(kf.split(x_train)):
            x_tr = x_train[tr]
            x_vl = x_train[val]
    
            y_tr = y_train[tr]
            y_vl = y_train[val]

            K.clear_session()
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

            rhos.append(cb.best_rho)
            models.append(model)

            if x_test is not None:
                test_preds.append(model.predict(x_test))

        return rhos, test_preds

    @classmethod
    def default_params(cls):
        params = {
            'temp_dir': Path('../.tmp'),
            'model_dir': Path('../model'),
            'maxlen': 512, 
            'seed': 42,
            'epochs': 100, 
            'bs': 32, 
            'n_splits': 5
        }
        return params