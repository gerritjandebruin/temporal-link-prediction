import collections, os, json, shutil, typing

import joblib
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
from tqdm.auto import tqdm

import tlp

def get_networks(dropna=True):
  networks = {
    1 : {'label': 'DBLP' , 'old_category': 'Coauthorship' , 'source': 'Ley2002'},
    2 : {'label': 'HepPh', 'old_category': 'Cocitation'   , 'source': 'Leskovec2007'},
    3 : {'label': 'Enron', 'old_category': 'Communication', 'source': 'Klimt2004'},
    4 : {'label': 'FB-w' , 'old_category': 'Social'       , 'source': 'Viswanath2009'},
    5 : {'label': 'Condm', 'old_category': 'Coauthorship' , 'source': 'Lichtenwalter2010'},
    6 : {'label': 'HepTh', 'old_category': 'Cocitation'   , 'source': 'Leskovec2007'},
    7 : {'label': 'AMin' , 'old_category': 'Coauthorship' , 'source': 'Zhuang2013'},
    8 : {'label': 'FB-l' , 'old_category': 'Social'       , 'source': 'Viswanath2009'},
    9 : {'label': 'D-rep', 'old_category': 'Communication', 'source': 'DeChoudhury2009'},
    10: {'label': 'D-f'  , 'old_category': 'Social'       , 'source': 'Hogg2010'},
    11: {'label': 'D-v'  , 'old_category': 'Rating'       , 'source': 'Hogg2010'},
    12: {'label': 'Rado' , 'old_category': 'Communication', 'source': 'Michalski2011'},
    13: {'label': 'UC'   , 'old_category': 'Interaction'  , 'source': 'Opsahl2013'},
    14: {'label': 'SX-MO', 'old_category': 'OnlineContact', 'source': 'Paranjape2017'},
    15: {                  'old_category': 'Social'       , 'source': ''},
    16: {'label': 'trust', 'old_category': 'Social'       , 'source': 'Richardson2003'},
    17: {                  'old_category': 'Social'       , 'source': ''},
    18: {'label': 'bitA' , 'old_category': 'Social'       , 'source': 'Kumar2017'},
    19: {'label': 'Dem'  , 'old_category': 'Social'       , 'source': 'Wikileaks'},
    20: {'label': 'bitOT', 'old_category': 'Coauthorship' , 'source': 'Kumar2017'},
    21: {'label': 'chess', 'old_category': 'Interaction'  , 'source': 'konect'},
    22: {'label': 'SX-AU', 'old_category': 'OnlineContact', 'source': 'Paranjape2017'},
    23: {'label': 'SX-SU', 'old_category': 'OnlineContact', 'source': 'Paranjape2017'},
    24: {'label': 'loans', 'old_category': 'Interaction'  , 'source': 'Redmond2013'},
    25: {'label': 'Wiki' , 'old_category': 'OnlineContact', 'source': 'Brandes2009'},
    26: {                  'old_category': 'Communication', 'source': 'Sun2016'},
    27: {                  'old_category': 'Hyperlink'    , 'source': 'Mislove2009'},
    28: {'label': 'Rbody', 'old_category': 'Hyperlink'    , 'source': 'Kumar2018'},
    29: {'label': 'Rtit' , 'old_category': 'Hyperlink'    , 'source': 'Kumar2018'},
    30: {'label': 'EU'   , 'old_category': 'Communication', 'source': 'Yin2017'},
  }

  for network_index, network_info in networks.items():
    old_category = network_info['old_category']
    if old_category in ['Coauthorship', 'Communication', 'OnlineContact']:
      network_info['category'] = 'social'
    elif old_category in ['Cocitation', 'Rating', 'Interaction']:
      network_info['category'] = 'information'
    elif old_category == 'Hyperlink':
      network_info['category'] = 'technological'
    else:
      network_info['category'] = old_category.lower()
  networks = pd.DataFrame.from_dict(networks, orient='index')
  if dropna: networks.dropna(inplace=True)
  return networks

exclude_indices = [15, 17, 26, 27]
network_indices = [
  network_index 
  for network_index in range(1, 31) if network_index not in exclude_indices]
network_count = len(network_indices)
hypergraph_indices = {
  1, 2, 3, 5, 6, 7, 12, 13, 14, 19, 22, 23, 25, 26, 28, 29, 30}
hypergraph_indices = {
  network_index 
  for network_index in hypergraph_indices 
  if network_index not in exclude_indices}
simplegraph_indices = {4, 8, 9, 10, 11, 15, 16, 17, 18, 20, 21, 24, 27}
simplegraph_indices = {
  network_index 
  for network_index in simplegraph_indices 
  if network_index not in exclude_indices}
networks = get_networks(dropna=True)
heuristics = ['aa', 'cn', 'jc', 'pa']

time_strategies = tlp.TIME_STRATEGIES
aggregation_strategies = tlp.AGGREGATION_STRATEGIES
nodepair_strategies = tlp.NODEPAIR_STRATEGIES

rc = {'xtick.top': True, 'ytick.right': True, 'figure.figsize': (4, 4)}

def get_size(network_index: int):
  edgelist = pd.read_pickle(f'data/{network_index:02}/edgelist.pkl')
  graph = nx.from_pandas_edgelist(edgelist)
  hypergraph = nx.from_pandas_edgelist(edgelist, create_using=nx.MultiGraph)
  size_statistics = pd.Series({
    'nodes': graph.number_of_nodes(),
    'edges': graph.number_of_edges(),
    'events': hypergraph.number_of_edges()},
    name=network_index) 
  return size_statistics
  
def get_all_stats(network_indices=network_indices):
  stats = dict()
  for network_index in tqdm(network_indices):
    with open(f'data/{network_index:02}/stats.json') as file:
      stats[network_index] = json.load(file)
  stats = pd.DataFrame.from_dict(stats, orient='index')
  stats.rename(
    columns={
      'density (nx.Graph)': 'density', 
      'degree assortativity (nx.Graph)': 'degree assortativity'}, 
    inplace=True)
  return stats

def get_diameter(network_indices=network_indices):
  diameters = dict()
  for network_index in tqdm(network_indices):
    with open(f'data/{network_index:02}/diameter.int') as file:
      diameters[network_index] = int(file.read())
#   diameters = pd.Series(diameters, name='diameter').astype(int)
  return diameters