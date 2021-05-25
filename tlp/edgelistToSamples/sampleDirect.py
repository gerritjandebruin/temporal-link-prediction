import os, pickle

import networkx as nx


def sample_direct(path, cutoff, verbose=False):
  filepath = os.path.join(path, 'edgelist_mature.pkl')
  if not os.path.isfile(filepath): return
  with open(filepath, 'rb') as file:
    edgelist_mature = pickle.load(file)
  graph_mature = nx.from_pandas_edgelist(edgelist_mature)