import pandas as pd

import tlp

def get_data() -> pd.DataFrame:
  df = pd.DataFrame.from_dict(tlp.stats.read_cheap_statistics(), orient='index')
  df['diameter'] = pd.Series(tlp.stats.get_diameter())
  df['label'] = pd.Series(tlp.recursive_file_loading('label.txt'))
  df['categories'] = pd.Series(tlp.get_categories())
  df['score'] = pd.Series(tlp.read_auc_from_logistic_regression())
  df['score time-agnostic'] = pd.Series(
    tlp.read_auc_from_logistic_regression('auc_time_agnostic.float')
  )
  
  df.reset_index(inplace=True)

  df.rename(
    columns={
      'density (nx.Graph)': 'd.', 
      'degree assortativity (nx.Graph)': 'd.a.'}, 
    inplace=True)
  
  return df