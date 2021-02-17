import collections.abc
import os

import joblib
import networkx as nx
import pandas as pd

from .experiment import Experiment

def aa_time_agnostic(edgelist: pd.DataFrame, 
                     instances: collections.abc.Iterable, *,
                     output_path: str,
                     ) -> None:
  """Calculate the time agnostic Adamic Adar feature.
  
  Args:
    edgelist: A pd.DataFrame with at least columns 'source' and 'target'.
    instances: An iterable yielding pairs of nodes.
    output_path: If provided, store the result at path/AA_time_agnostic.pkl.
  
  Stores at os.path.join(output_path, 'AA_time_agnostic.pkl'):
    A dict with as key a NamedTuple (Experiment) and as value a np.array
      containing the scores.
  """
  file = os.path.join(output_path, 'AA_time_agnostic.pkl')
  os.makedirs(output_path, exist_ok=True)
  if not os.path.isfile(file):
    result = {
      Experiment('AA', time_aware=False): (
        pd.Series(
          [p 
          for u, v, p 
          in nx.adamic_adar_index(G=nx.from_pandas_edgelist(edgelist), 
                                  ebunch=instances)]))}
    joblib.dump(result, file)
