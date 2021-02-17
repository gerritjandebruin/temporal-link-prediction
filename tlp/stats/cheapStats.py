import datetime
import json
import os

import joblib
import networkx as nx

def _gc(g: nx.Graph) -> nx.Graph:
  """Return giant component."""
  return g.subgraph(max(nx.connected_components(g), key=len)).copy()

def cheap_statistics(path: str, verbose: bool = False) -> None:
  """Print some statistics that are relatively cheap to compute.
  This method is made for graphs that have multiple edges between
  nodes (nx.MultiGraph).

  Args:
    path: In this path, edgelist.pkl should be present.
    verbose: Optional; Defaults to False.
  """
  result = dict()
  
  if verbose: print(f'{datetime.datetime.now()} Get edgelist')
  edgelist = joblib.load(os.path.join(path, 'edgelist.pkl'))
  
  if verbose: print(f'{datetime.datetime.now()} Get nx.MultiGraph')
  multigraph = nx.from_pandas_edgelist(edgelist, create_using=nx.MultiGraph)
  result['edges'] = multigraph.number_of_edges()
  # if verbose: print(f'{datetime.datetime.now()} Get degree assortativity')
  # result['degree assortativity (nx.MultiGraph)'] = (
  #   nx.degree_assortativity_coefficient(multigraph))
  if verbose: print(f'{datetime.datetime.now()} Get density')
  result['density (nx.MultiGraph)'] = nx.density(multigraph)
  
  if verbose: print(f'{datetime.datetime.now()} Get GC of nx.MultiGraph')
  multigraph_gc = _gc(multigraph)
  del multigraph
  multigraph_gc_number_of_edges = multigraph_gc.number_of_edges()
  result['fraction edges in GC'] = (
    multigraph_gc_number_of_edges / result['edges'])
  del multigraph_gc
  
  if verbose: print(f'{datetime.datetime.now()} Get nx.Graph')
  simplegraph = nx.from_pandas_edgelist(edgelist)
  del edgelist
  result['nodes'] = simplegraph.number_of_nodes()
  result['avg events per pair'] = (
    result['edges'] / simplegraph.number_of_edges())
  if verbose: print(f'{datetime.datetime.now()} Get density')
  result['density (nx.Graph)'] = nx.density(simplegraph)
  if verbose: print(f'{datetime.datetime.now()} Get degree assortativity')
  result['degree assortativity (nx.Graph)'] = (
    nx.degree_assortativity_coefficient(simplegraph))
  if verbose: print(f'{datetime.datetime.now()} Get clustering coefficient')
  result['average clustering coefficient'] = (
    nx.average_clustering(simplegraph))
  
  if verbose: print(f'{datetime.datetime.now()} Get GC of nx.Graph')
  simplegraph_gc = _gc(simplegraph)
  del simplegraph
  result['fraction nodes in GC'] = (
    simplegraph_gc.number_of_nodes() / result['nodes'])
  result['avg events per pair in GC'] = (
    multigraph_gc_number_of_edges / simplegraph_gc.number_of_edges())
  
  if verbose: print(f'{datetime.datetime.now()} Store results')
  with open(os.path.join(path, 'stats.json'), 'w') as file:
    json.dump(result, file)