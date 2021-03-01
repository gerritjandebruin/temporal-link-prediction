import collections, os, typing

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
from tqdm.auto import tqdm

import tlp

feature = 'ra'
time_strategies = tlp.TIME_STRATEGIES
aggregation_strategies = tlp.AGGREGATION_STRATEGIES

hypergraph = {1, 2, 3, 5, 6, 7, 12, 13, 14, 19, 22, 23, 25, 26, 28, 29, 30}
simplegraph = {4, 8, 9, 10, 11, 15, 16, 17, 18, 20, 21, 24, 27}

def ra(index, time_str, time_func, agg_str=None, agg_func=None, feature=feature):
  # Check if file already exists
  result_file = os.path.join('data', f'{index:02}', 'features', 'time_edge', f'{feature}_{time_str}.npy' if agg_str is None else f'{feature}_{time_str}_{agg_str}.npy')
  if os.path.isfile(result_file): return
  
  # Read in
  edgelist_mature = pd.read_pickle(os.path.join('data', f'{index:02}', 'edgelist_mature.pkl'))
  instances = np.load(os.path.join(os.path.join('data', f'{index:02}', 'instances_sampled.npy')))
  
  # nodes = {node for instance in instances_sampled for node in instance}
  
  # Apply time strategy
  edgelist_mature['datetime_transformed'] = time_func(edgelist_mature['datetime'])

  # Create multigraph
  G = nx.from_pandas_edgelist(edgelist_mature, create_using=nx.MultiGraph, edge_attr=True)
  
  if agg_func is None:
    agg_func = np.max

  # Calculation
  scores = [
    sum(
      [
        (
          agg_func([edge_attributes['datetime_transformed'] for edge_attributes in G.get_edge_data(u, z).values()]) +
          agg_func([edge_attributes['datetime_transformed'] for edge_attributes in G.get_edge_data(v, z).values()])
        ) / sum([agg_func([edge_attributes['datetime_transformed'] for edge_attributes in G.get_edge_data(c, z).values()]) for c in G[z]])
        for z in nx.common_neighbors(G, u, v)
      ]
    )
    for u, v in instances
  ]
  np.save(result_file, scores)

def main():
  np.seterr('raise')

  args = [
    dict(index=index, time_str=time_str, time_func=time_func, agg_str=agg_str, agg_func=agg_func)
    for index in hypergraph
    for time_str, time_func in time_strategies.items()
    for agg_str, agg_func in aggregation_strategies.items()
  ] + [
    dict(index=index, time_str=time_str, time_func=time_func)
    for index in simplegraph
    for time_str, time_func in time_strategies.items()
  ]
  total = len(args)
  # tlp.ProgressParallel(n_jobs=min(50, total), total=total)(
  #   joblib.delayed(ra)(**arg) 
  #   for arg in args
  # )
  for arg in tqdm(args):
    ra(**arg)

if __name__ == '__main__':
  main()