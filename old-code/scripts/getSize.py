import joblib
import pandas as pd

def main(network: int):
  edgelist = pd.read_pickle(f'data/{network:02}/edgelist.pkl')
  graph = nx.from_pandas_edgelist(edgelist, create_using=nx.MultiGraph)
  with open(f'data/{network:02}/nodes.int', 'w') as file:
    file.write(str(graph.number_of_nodes()))
  with open(f'data/{network:02}/edges.int', 'w') as file:
    file.write(str(graph.number_of_edges()))
  with open(f'data/{network:02}/density.float', 'w') as file:
    file.write(str(nx.density(nx.Graph(graph))))
    
if __name__ == '__main__':
  networks = [int(network) for network in os.listdir('data')]
  joblib.Parallel(n_jobs=len(networks))(
    joblib.delayed()
  )