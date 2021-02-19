import os

import joblib
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing

from .helpers import recursive_file_loading

def calculate_auc_from_logistic_regression(index: int) -> None:
  X = dict()
  
  data_dir = os.path.join('data', f'{index:02}')
  result_file = os.path.join(data_dir, 'scores', 'auc_all_features.float')

  # If result is already calculated, quit.
  if os.path.isfile(result_file): return

  # If features are not calculated yet, quit.
  if not os.path.isdir(os.path.join(data_dir, 'features')): return

  # Get all calculated features.
  for file in os.scandir(os.path.join(data_dir, 'features')):
    X.update(joblib.load(file.path))
  X = pd.DataFrame(X)

  # Get targets
  y = np.load(os.path.join(data_dir, 'targets_sampled.npy'))

  # Fit and predict pipeline
  X_train, X_test, y_train, y_test = (
    sklearn.model_selection.train_test_split(X, y))
  pipe = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    sklearn.linear_model.LogisticRegression(max_iter=10000)) # type: ignore
  pipe.fit(X_train, y_train)
  auc = sklearn.metrics.roc_auc_score(
    y_true=y_test, y_score=pipe.predict_proba(X_test)[:,1]) # type: ignore

  # Store result
  os.makedirs(os.path.join(data_dir, 'scores'), exist_ok=True)
  with open(result_file, 'w') as file:
    file.write(f'{auc:.2}')  

def read_auc_from_logistic_regression():
  return recursive_file_loading('scores/auc_all_features.float')
