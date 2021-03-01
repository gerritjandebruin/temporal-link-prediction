import os
import subprocess
import typing

import joblib
import networkx as nx
import numpy as np

from ..helpers import print_status, recursive_file_loading

def get_path_distribution(
  path: str, sample_size: typing.Optional[int] = None
) -> None:
  result_file = os.path.join(path, 'path_distribution.npy')
  if os.path.isfile(result_file): return None

  numeric_edgelist_path = os.path.join(path, 'numeric_edgelist.tsv')
  edgelist = joblib.load(os.path.join(path, 'edgelist.pkl'))
  graph = nx.from_pandas_edgelist(edgelist)

  if not os.path.isfile(numeric_edgelist_path):
    graph = nx.convert_node_labels_to_integers(graph)
    nx.write_edgelist(graph, numeric_edgelist_path, delimiter='\t', data=False)
  
  if sample_size is not None:
    number_of_nodes = graph.number_of_nodes()
    print(f'{number_of_nodes=}')
    if number_of_nodes < sample_size:
      sample_size = None

  if sample_size is None:
    input = f'load_undirected {numeric_edgelist_path}\ndist_distri'
  else:
    input = (
      f'load_undirected {numeric_edgelist_path}\n'
      f'est_dist_distri {sample_size / number_of_nodes}') #type: ignore

  teexgraph_process = subprocess.run(
    ['./teexgraph/teexgraph'], 
    input=input, 
    encoding='ascii',
    stdout=subprocess.PIPE
  )
  
  print(f'Return code: {teexgraph_process.returncode}')

  if teexgraph_process.returncode == 0:
    path_distribution = [
      int(item)
      for line in teexgraph_process.stdout.split('\n')
      for item in line.split('\t') if item != ''
    ]
    
    path_distribution = np.array(path_distribution).reshape(-1,2)
    path_distribution_file = os.path.join(path, 'path_distribution')
    np.save(path_distribution_file, path_distribution)

def read_path_distribution():
  results = recursive_file_loading('path_distribution.npy')
  
  for index, result in results.items():
    if result.shape[0] == 0:
      print_status(f'#{index:02} failed: empty np.array')
  
  return {
    index: result for index, result in results.items() if result.shape[0] > 0}

def get_average_shortest_simple_path_length():
  return {
    index: np.average(path_distribution[:,0], weights=path_distribution[:,1]) 
    for index, path_distribution in read_path_distribution().items()
  }