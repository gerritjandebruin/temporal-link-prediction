import os
import pickle
import shutil

import numpy as np

if __name__ == '__main__':
  for root, dirs, files in os.walk('data/'):
    for file in ['PR.pkl']:
      if file in files:
        os.remove(os.path.join(root, file))
    # if 'features' in dirs:
    #   shutil.rmtree(os.path.join(root, 'features'))
      
