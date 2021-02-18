import os
import typing

def file_exists(files: typing.Union[str, list[str]], *, verbose: bool = False
                ) -> bool:
  """Check if file (or files) exists. If any exists, return True."""
  if type(files) == str:
    files = list(files) # ignore: type
    
  for file in files:
    if os.path.isfile(file):
      if verbose: print(f"{file} already exists")
      return True
      
  return False