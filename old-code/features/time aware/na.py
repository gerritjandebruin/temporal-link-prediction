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

time_strategies = tlp.TIME_STRATEGIES
aggregation_strategies = tlp.AGGREGATION_STRATEGIES
nodepair_strategies = tlp.NODEPAIR_STRATEGIES

hypergraph = {1, 2, 3, 5, 6, 7, 12, 13, 14, 19, 22, 23, 25, 26, 28, 29, 30}
simplegraph = {4, 8, 9, 10, 11, 15, 16, 17, 18, 20, 21, 24, 27}

def na(index, time_str, time_func, nodepair_str, nodepair_func, agg_str=None, agg_func=None):
  # Check if file already exists
  result_file = os.path.join('data', f'{index:02}', 'features', 'time_edge', f'{time_str}_{nodepair_str}.npy' if agg_str is None else f'{time_str}_{nodepair_str}_{agg_str}.npy')
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
    nodepair_func(
      [
        agg_func([datetime for _, _, datetime in G.edges(u, data='datetime_transformed')]),
        agg_func([datetime for _, _, datetime in G.edges(v, data='datetime_transformed')])
      ]
    )
    for u, v in instances
  ]
  np.save(result_file, scores)

def main():
  args = [
    dict(index=index, time_str=time_str, time_func=time_func, agg_str=agg_str, agg_func=agg_func, nodepair_str=nodepair_str, nodepair_func=nodepair_func)
    for index in hypergraph
    for time_str, time_func in time_strategies.items()
    for nodepair_str, nodepair_func in nodepair_strategies.items()
    for agg_str, agg_func in aggregation_strategies.items()
  ] + [
    dict(index=index, time_str=time_str, time_func=time_func, nodepair_str=nodepair_str, nodepair_func=nodepair_func)
    for index in simplegraph
    for time_str, time_func in time_strategies.items()
    for nodepair_str, nodepair_func in nodepair_strategies.items()
  ]
  total = len(args)
  tlp.ProgressParallel(n_jobs=50, total=total)(
    joblib.delayed(na)(**arg) 
    for arg in args
    if arg['index'] != 15
  )
  # for arg in tqdm(args):
  #   na(**arg)

if __name__ == '__main__':
  main()