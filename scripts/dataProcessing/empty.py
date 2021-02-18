make_undirected = {12: False}
make_undirected = collections.defaultdict(lambda: True, make_undirected)

adjusted_intervals = {
  1: {
      't_min': pd.Timestamp('1996-01-01'), 
      't_split': pd.Timestamp('2005-01-01'),
      't_max': pd.Timestamp('2007-01-01')
    },
  16: {'t_min': pd.Timestamp('2001-01-10 00:00:00')}
}