import tensorflow as tf
import numpy as np
import random

def go_deterministic(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)