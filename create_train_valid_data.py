import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold

seed = 42

train_df = pd.read_csv('data/train.csv')

ids = np.arange(len(train_df))

kf = KFold(n_splits=5, shuffle=True, random_state=seed).split(ids, groups=train_df.question_body)

for i, (tr_ids, val_ids) in enumerate(kf):
    pd.DataFrame(data={'ids': tr_ids}).to_csv(f"data/train_ids_fold_{i}.csv", index=False)
    pd.DataFrame(data={'ids': val_ids}).to_csv(f"data/valid_ids_fold_{i}.csv", index=False)