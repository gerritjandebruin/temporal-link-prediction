import os
import subprocess
import tempfile
import typing

import joblib
import networkx as nx
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
import numpy as np

def path_distribution(path: str, sample_size: typing.Optional[int] = 2000
) -> None:
  edgelist_path = os.path.join(path, 'edgelist.pkl')
  edgelist = joblib.load(edgelist_path)
  graph = nx.from_pandas_edgelist(edgelist)

  graph = nx.convert_node_labels_to_integers(graph)

  numeric_edgelist_path = os.path.join(path, 'numeric_edgelist.tsv')
  nx.write_edgelist(graph, numeric_edgelist_path, delimiter='\t', data=False)

  if sample_size is None:
    input = 'load_undirected {numeric_edgelist_path}\ndist_distri'
  else:
    sample_size = sample_size / graph.number_of_nodes()
    input = (
      f'load_undirected {numeric_edgelist_path}\nest_dist_distri {sample_size}')

  teexgraph_process = subprocess.run(
    ['./teexgraph/teexgraph'], 
    input=input, 
    encoding='ascii',
    stdout=subprocess.PIPE
  )
  
  path_distribution = [
    int(item)
    for line in teexgraph_process.stdout.split('\n')
    for item in line.split('\t') if item != ''
  ]
  path_distribution = np.array(path_distribution).reshape(-1,2)
  path_distribution_file = os.path.join(path, 'path_distribution')
  np.save(path_distribution_file, path_distribution)

  average_shortest_simple_path_length = (
    np.average(path_distribution[:,0], weights=path_distribution[:,1]))
  output_file = os.path.join('average_shortest_path_length.txt')
  with open(output_file, 'w') as file:
    file.write(str(average_shortest_simple_path_length))