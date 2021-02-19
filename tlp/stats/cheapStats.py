import datetime
import json
import os

import joblib
import networkx as nx

from ..helpers import print_status, recursive_file_loading

def _gc(g: nx.Graph) -> nx.Graph:
  """Return giant component."""
  return g.subgraph(max(nx.connected_components(g), key=len)).copy()

def calculate_cheap_statistics(index: int, verbose: bool = False) -> None:
  """Print some statistics that are relatively cheap to compute.
  This method is made for graphs that have multiple edges between
  nodes (nx.MultiGraph).

  Args:
    path: In this path, edgelist.pkl should be present.
    verbose: Optional; Defaults to False.
  """
  result = dict()
  path = os.path.join('data', f'{index:02}')
  
  if verbose: print_status(f'#{index:02}: Get edgelist')
  edgelist = joblib.load(os.path.join(path, 'edgelist.pkl'))
  
  if verbose: print_status(f'#{index:02}: Get nx.MultiGraph')
  multigraph = nx.from_pandas_edgelist(edgelist, create_using=nx.MultiGraph)
  result['edges'] = multigraph.number_of_edges()
  # if verbose: print_status(f'#{index:02}: Get degree assortativity')
  # result['degree assortativity (nx.MultiGraph)'] = (
  #   nx.degree_assortativity_coefficient(multigraph))
  if verbose: print_status(f'#{index:02}: Get density')
  result['density (nx.MultiGraph)'] = nx.density(multigraph)
  
  if verbose: print_status(f'#{index:02}: Get GC of nx.MultiGraph')
  multigraph_gc = _gc(multigraph)
  del multigraph
  multigraph_gc_number_of_edges = multigraph_gc.number_of_edges()
  result['fraction edges in GC'] = (
    multigraph_gc_number_of_edges / result['edges'])
  del multigraph_gc
  
  if verbose: print_status(f'#{index:02}: Get nx.Graph')
  simplegraph = nx.from_pandas_edgelist(edgelist)
  del edgelist
  result['nodes'] = simplegraph.number_of_nodes()
  result['avg events per pair'] = (
    result['edges'] / simplegraph.number_of_edges())
  if verbose: print_status(f'#{index:02}: Get density')
  result['density (nx.Graph)'] = nx.density(simplegraph)
  if verbose: print_status(f'#{index:02}: Get degree assortativity')
  result['degree assortativity (nx.Graph)'] = (
    nx.degree_assortativity_coefficient(simplegraph))
  if verbose: print_status(f'#{index:02}: Get clustering coefficient')
  result['average clustering coefficient'] = (
    nx.average_clustering(simplegraph))
  
  if verbose: print_status(f'#{index:02}: Get GC of nx.Graph')
  simplegraph_gc = _gc(simplegraph)
  del simplegraph
  result['fraction nodes in GC'] = (
    simplegraph_gc.number_of_nodes() / result['nodes'])
  result['avg events per pair in GC'] = (
    multigraph_gc_number_of_edges / simplegraph_gc.number_of_edges())
  
  if verbose: print_status(f'#{index:02}: Store results')
  with open(os.path.join(path, 'stats.json'), 'w') as file:
    json.dump(result, file)

def read_cheap_statistics() -> dict[int, dict]:
  return recursive_file_loading('stats.json')