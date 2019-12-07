
from functools import partial

import tensorflow as tf
import numpy as np

def apply_tokenizer(tokenizer, maxlen, text):
    text = ' '.join(text.numpy().decode('utf-8').strip().split(' ')[:maxlen])
    tokens = np.zeros(maxlen, dtype=np.int32)
    text_tokens = tokenizer.encode(text, max_length=maxlen, add_special_tokens=True)[:maxlen]
    tokens[:len(text_tokens)] = text_tokens
    attention_mask = np.where(tokens != 0, 1, 0)  
    return tokens, attention_mask

def tf_apply_tokenizer(tokenizer, maxlen, text):
    f = partial(apply_tokenizer, tokenizer, maxlen)
    tokens, attention_mask = tf.py_function(f, [text], [tf.int32, tf.int32])
    
    tokens.set_shape(maxlen)
    attention_mask.set_shape(maxlen)
    
    return (tokens,attention_mask)

def get_features(model, tokenizer, maxlen, texts):
    f = partial(tf_apply_tokenizer, tokenizer, maxlen)
    dataset = tf.data.Dataset.from_tensor_slices((texts)).map(f).batch(8)
    
    inputs = tf.keras.layers.Input(shape=(maxlen), dtype=tf.int32)
    x = model(inputs)[0][:,0,:]
    
    pooled = tf.keras.Model(inputs=inputs, outputs=x)
    
    outs = pooled.predict(dataset)
    # ref: https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/transformers/modeling_tf_distilbert.py#L697
    return outs