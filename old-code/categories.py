import datetime
import json
import os
import pickle
import re
import typing

import joblib
import numpy as np
from tqdm.auto import tqdm

def get_categories():
  result = {
    5: 'Coauthorship',
    7: 'Coauthorship',
    28: 'Hyperlink',
    29: 'Hyperlink',
    30: 'Communication',
  }
  entries = sorted(os.scandir('data'), key=lambda x: x.name)
  for entry in entries:
    meta_file = os.path.join(entry.path, 'meta')
    if os.path.isfile(meta_file):
      with open(meta_file) as file:
        for line in file:
          if line.startswith('category'):
            result[int(entry.name)] = line.split(':')[1].strip()
  return result