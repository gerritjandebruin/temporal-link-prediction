from pandas.core.frame import DataFrame
from getData import get_data # type: ignore

if __name__ == "__main__":
  df = get_data()[['label', 'categories', 'nodes', 'd.', 'd.a.']]
  df.sort_values(['categories', 'label'], inplace=True)
  df.reset_index(drop=True, inplace=True)
  print(
    df.to_latex(
      formatters={'⌀': lambda x: f'{x:.0e}', 'd.a.': lambda x: f'{x:.1e}'}, 
      caption='Datasets used in this work. d. indicates diameter of the network'
        ' and d.a. the degree assortativity.',
      label='table:datasets')
  )