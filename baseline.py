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
from constants import text_columns

import pdb

class Baseline():

    def __init__(self, params):
        self.params = params
        self.features = None  

    def _get_bert_hidden_states(self, model, tokenizer, df):
        question_body = get_features(model, tokenizer, self.params['maxlen'], df.question_body.values)
        answer = get_features(model, tokenizer, self.params['maxlen'], df.answer.values)
        question_title = get_features(model, tokenizer, self.params['maxlen'], df.question_title.values)
        return np.hstack([question_body, answer, question_title])

    def load_features(self):
        with open(self.params['temp_dir'] / 'features.pickle', 'rb') as f:
            self.features = pickle.load(f)

    def calculate_features(self, data: Dict[str, pd.DataFrame], save=False):
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

    def train_and_evaluate(self):
        pass

    @classmethod
    def default_params(cls):
        params = {
            'temp_dir': Path('../.tmp'),
            'model_dir': Path('../model'),
            'maxlen': 512
        }
        return params