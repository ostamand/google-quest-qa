import numpy as np
import pandas as pd 

from typing import List, Dict

l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)
cos_dist = lambda x, y: (x*y).sum(axis=1)

def get_dist_features(embeddings: Dict[str, np.array]):
    """Calculate distance features from embeddings dictionary."""
    dist = np.array([
        l2_dist(embeddings['question_title_embeds'], embeddings['answer_embeds']),
        l2_dist(embeddings['question_body_embeds'], embeddings['answer_embeds']),
        l2_dist(embeddings['question_body_embeds'], embeddings['question_title_embeds']),
        cos_dist(embeddings['question_title_embeds'], embeddings['answer_embeds']),
        cos_dist(embeddings['question_body_embeds'], embeddings['answer_embeds']),
        cos_dist(embeddings['question_body_embeds'], embeddings['question_title_embeds'])
        ]).T
    return dist 

def embeddings_from_col(df: pd.DataFrame, cols: List, model, **kwargs):
    embeddings = {}
    for col in cols: 
        text = df[col].values
        # TODO replace ! ? by .
        embeddings[col + '_embeds'] = get_sentence_embeddings(text, model, **kwargs)
    return embeddings

def get_sentence_embeddings(text, model, **kwargs):
    bs = kwargs['bs'] if 'bs' in kwargs else 32
    embeddings = []
    for i in range(int(np.ceil(len(text)/bs))):
        embeddings.append(model(text[i*bs:(i+1)*bs]).numpy())
    embeddings = np.vstack(embeddings)
    return embeddings