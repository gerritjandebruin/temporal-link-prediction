import itertools, os, random

import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline
import typer
import xgboost as xgb
from tqdm.auto import tqdm

from .progress_parallel import ProgressParallel, delayed

app = typer.Typer()

def predict(directory: str, 
            feature_set='II-A',
            clf='LogisticRegression',
            random_state=42, 
            n_jobs=-1):
    if feature_set == 'I':
        check = lambda x: x in ['aa.npy', 'cn.npy', 'jc.npy', 'pa.npy', 'sp.npy']
    elif feature_set == 'II-A':
        check = lambda x: not x.startswith('na') and not 'm5' in x
    elif feature_set == 'II-B':
        check = lambda x: (
            x in ['aa.npy', 'cn.npy', 'jc.npy', 'pa.npy', 'sp.npy'] or
            ('_q100' in x) and not x.startswith('na') and not 'm5' in x
        )
    elif feature_set == 'III':
        check = lambda x: (
            x in ['aa.npy', 'cn.npy', 'jc.npy', 'pa.npy', 'sp.npy'] or 
            x.startswith('na') and not 'm5' in x
        )
    else:
        raise Exception(f'{feature_set=} not recognized')
  
    assert os.path.isdir(directory), f'missing {directory=}'
    feature_dir = os.path.join(directory, 'features')
    if not os.path.isdir(feature_dir):
        return None
    samples_filepath = os.path.join(directory, 'samples.pkl')
    assert os.path.isfile(samples_filepath), f'missing {samples_filepath=}'
  
    X = pd.DataFrame({
        f.name: np.load(f.path) 
        for f in os.scandir(feature_dir) if check(f.name)
    })
    
    y = pd.read_pickle(samples_filepath).astype(int).values
  
    X_train, X_test, y_train, y_test = (
        sklearn.model_selection.train_test_split(X, y, random_state=random_state))
    if clf == 'LogisticRegression':
        pipe = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.LogisticRegression(max_iter=10000, n_jobs=n_jobs, 
                                                    random_state=random_state)
        )
    elif clf == 'RandomForest':
        pipe = sklearn.pipeline.make_pipeline(
            sklearn.ensemble.RandomForestClassifier(
                random_state=random_state, n_jobs=100 if n_jobs < 0 else n_jobs
            )
        )
    elif clf == 'XGBoost':
        pipe = xgb.XGBClassifier(n_jobs=100 if n_jobs < 0 else n_jobs, 
                                 random_state=random_state,
                                 use_label_encoder=False)
        pipe.fit(X_train, y_train, eval_metric='logloss')
    else:
        raise Exception(f'Invalid clf argument: {clf}')
    
    if not clf == 'XGBoost':
        pipe.fit(X_train, y_train)   
  
    auc = sklearn.metrics.roc_auc_score(
        y_true=y_test, y_score=pipe.predict_proba(X_test)[:,1])
  
    return auc

@app.command()
def single(network: int, 
           nswap_perc: int, 
           method: str = None,
           clf: str = 'LogisticRegression',
           feature_set: str = 'II-A',
           random_state: int = 42,
           n_jobs: int = -1):
    assert nswap_perc != 0 or method is None, f'got {nswap_perc=} and {method=}'
    if nswap_perc == 0:
        directory = f'data/{network:02}/{nswap_perc:+04.0f}'
    else:
        directory = f'data/{network:02}/{nswap_perc:+04.0f}{method}'
    os.makedirs(directory, exist_ok=True)
    filepath_out = os.path.join(directory, 'properties', 
                                f'{feature_set}_{clf}.float')
    if os.path.isfile(filepath_out):
        return
    auc = predict(directory, feature_set, clf, random_state, n_jobs)
    if auc is not None:
        with open(filepath_out, 'w') as file:
            file.write(str(auc))
      
  
@app.command()
def all(network: int = None,
        method: str = None,
        clf: str = 'LogisticRegression',
        n_jobs: int = -1, 
        shuffle: bool = True, 
        seed: int = 42,
        include_nswap: bool = True):
    if network is None:
        networks = [network for network in np.arange(1, 31) 
                    if network not in [15, 17, 26, 27]]
    else:
        networks = [network]
    if include_nswap:
        nswap_percs = np.arange(-100, 101, 20)
    else:
        nswap_percs = [0]
    if method is None:
        iterations = [(network, nswap_perc, None if nswap_perc == 0 else method)
                      for network in networks
                      for nswap_perc in nswap_percs
                      for method in ['a', 'b']
                      if not nswap_perc == 0 and method == 'b']
    else:
        assert method in ['a', 'b']
        iterations = [(network, nswap_perc, None if nswap_perc == 0 else method)
                      for network in networks
                      for nswap_perc in nswap_percs]
    if shuffle:
        random.seed(seed)
        random.shuffle(iterations)
    if n_jobs == -1 or n_jobs > 1:
        ProgressParallel(n_jobs=n_jobs, total=len(iterations))(
            delayed(single)(
                network=network, 
                method=method,
                clf=clf,
                nswap_perc=nswap_perc) 
            for network, nswap_perc, method in iterations
        )          
    else:
        for network, nswap_perc, method in tqdm(iterations):
            single(network, nswap_perc, method)
        
if __name__ == '__main__':
    app()
