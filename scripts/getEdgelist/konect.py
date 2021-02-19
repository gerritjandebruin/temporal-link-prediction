import collections

import joblib
from tqdm.auto import tqdm

import tlp

urls = {
  1: 'http://konect.cc/files/download.tsv.dblp_coauthor.tar.bz2',
  2: 'http://konect.cc/files/download.tsv.ca-cit-HepPh.tar.bz2', 
  3: 'http://konect.cc/files/download.tsv.enron.tar.bz2',
  4: 'http://konect.cc/files/download.tsv.facebook-wosn-links.tar.bz2',
  6: 'http://konect.cc/files/download.tsv.ca-cit-HepTh.tar.bz2',
  8: 'http://konect.cc/files/download.tsv.facebook-wosn-links.tar.bz2',
  9: 'http://konect.cc/files/download.tsv.munmun_digg_reply.tar.bz2',
  10: 'http://konect.cc/files/download.tsv.digg-friends.tar.bz2',
  11: 'http://konect.cc/files/download.tsv.digg-votes.tar.bz2',
  12: 'http://konect.cc/files/download.tsv.radoslaw_email.tar.bz2',
  13: 'http://konect.cc/files/download.tsv.opsahl-ucforum.tar.bz2',
  14: 'http://konect.cc/files/download.tsv.sx-mathoverflow.tar.bz2',
  15: 'http://konect.cc/files/download.tsv.flickr-growth.tar.bz2',
  16: 'http://konect.cc/files/download.tsv.epinions.tar.bz2',
  17: 'http://konect.cc/files/download.tsv.youtube-u-growth.tar.bz2',
  18: 'http://konect.cc/files/download.tsv.soc-sign-bitcoinalpha.tar.bz2',
  19: 'http://konect.cc/files/download.tsv.dnc-temporalGraph.tar.bz2',
  20: 'http://konect.cc/files/download.tsv.soc-sign-bitcoinotc.tar.bz2',
  21: 'http://konect.cc/files/download.tsv.chess.tar.bz2',
  22: 'http://konect.cc/files/download.tsv.sx-askubuntu.tar.bz2',
  23: 'http://konect.cc/files/download.tsv.sx-superuser.tar.bz2',
  24: 'http://konect.cc/files/download.tsv.prosper-loans.tar.bz2',
  25: 'http://konect.cc/files/download.tsv.wikiconflict.tar.bz2',
  26: 'http://konect.cc/files/download.tsv.wiki_talk_en.tar.bz2',
  27: 'http://konect.cc/files/download.tsv.wikipedia-growth.tar.bz2',
}

if __name__ == "__main__":
  # Parallel
  # tlp.ProgressParallel(n_jobs=len(urls), total=len(urls))(
  #   joblib.delayed(tlp.get_edgelist_from_konect)
  #   (url, path=f'data/{index:02}') for index, url in urls.items()
  # )



  # For loop
  for index, url in tqdm(urls.items(), mininterval=0):
    tlp.get_edgelist_from_konect(url, path=f'data/{index:02}', verbose=True)