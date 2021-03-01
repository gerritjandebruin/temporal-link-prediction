import os
import subprocess
import tempfile

import joblib
import networkx as nx

def get_diameter(path: str) -> None:
  """Determine the diameter of the given graph. Result is stored at 
  path/diameter.int"""
  result_file = os.path.join(path, 'diameter.int')
  if os.path.isfile(result_file): return

  edgelist_path = os.path.join(path, 'edgelist.pkl')
  edgelist = joblib.load(edgelist_path)
  graph = nx.from_pandas_edgelist(edgelist)
  with tempfile.NamedTemporaryFile() as tmp:
    nx.write_edgelist(
      nx.convert_node_labels_to_integers(graph), 
      tmp.name, 
      delimiter='\t', 
      data=False
    )
    cp = subprocess.run(
      ['./teexgraph/teexgraph'], 
      input=f'load_undirected {tmp.name}\ndiameter', 
      encoding='ascii', 
      stdout=subprocess.PIPE
    )

  with open(result_file, 'w') as file:
    file.write(str(int(cp.stdout.split()[0])))
