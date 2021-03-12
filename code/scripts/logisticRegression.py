import os

from tqdm.auto import tqdm

import tlp

if __name__ == "__main__":
  indices = [
    int(index) for index in sorted(os.listdir('data')) 
    if not index.startswith('.')]

  # Singlecore
  for index in tqdm(indices):
    tlp.calculate_auc_from_logistic_regression(
      index, 
      filename='auc_time_agnostic.float', 
      features=['aa_time_agnostic.pkl', 'sp.pkl'])
    
  # Multicore
  # tlp.ProgressParallel(n_jobs=len(indices), total=len(indices))(
  #   joblib.delayed(tlp.get_auc_from_logistic_regression)(index) 
  #   for index in indices
  # )