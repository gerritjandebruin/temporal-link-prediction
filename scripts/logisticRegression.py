import os

from tqdm.auto import tqdm

import tlp

if __name__ == "__main__":
  indices = [int(index) for index in sorted(os.listdir('data'))]

  # Singlecore
  for index in tqdm(indices):
    tlp.get_auc_from_logistic_regression(index)
    
  # Multicore
  # tlp.ProgressParallel(n_jobs=len(indices), total=len(indices))(
  #   joblib.delayed(tlp.get_auc_from_logistic_regression)(index) 
  #   for index in indices
  # )