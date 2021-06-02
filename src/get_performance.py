import os

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline

def model_I(path: str):
  X = pd.DataFrame(
    {heuristic: np.load(os.path.join(path, 'features', heuristic + '.npy'))
     for heuristic in ['aa', 'cn', 'jc', 'pa']}
  )
  y = pd.Series(os.path.join(path, 'samples.pkl')).values
  
  X_train, X_test, y_train, y_test = (
    sklearn.model_selection.train_test_split(X, y, random_state=42))
  
  pipe = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    sklearn.linear_model.LogisticRegression(max_iter=10000, n_jobs=-1))
  pipe.fit(X_train, y_train)
  
  auc = sklearn.metrics.roc_auc_score(
    y_true=y_test, y_score=pipe.predict_proba(X_test)[:,1]
  )
  
  return auc