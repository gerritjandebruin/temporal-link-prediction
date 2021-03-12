import argparse
import os

from tqdm.auto import tqdm

import tlp.getEdgelist

konect_urls = {
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

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--check', help='Check if edgelist.pkl already exists.', 
    action='store_true')
  args = parser.parse_args()

  if args.check:
    ok = True
    for entry in os.scandir('data/'):
      if not os.path.isfile(os.path.join(entry.path, 'edgelist.pkl')):
        print(f'No edgelist.pkl in {entry.path}.')
        ok = False
    if ok:
      print('In all folders edgelist.pkl is present.')
    return()

  # KONECT
  for index, url in tqdm(konect_urls.items(), mininterval=0):
    tlp.getEdgelist.from_konect(url, path=f'data/{index:02}', verbose=True)

  # act-mooc (bipartite?)
  # tlp.getEdgelist.act_mooc(path='data/31', verbose=True)

  # AMiner
  tlp.getEdgelist.aminer(path='data/07', verbose=True)

  # email-Eu
  tlp.getEdgelist.email_EU(path='data/30', verbose=True)

  # reddit
  tlp.getEdgelist.reddit(
    path='data/28', 
    url='https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv', 
    verbose=True
  )
  tlp.getEdgelist.reddit(
    path='data/29', 
    url='https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv', 
    verbose=True
  )

if __name__ == '__main__':
  main()