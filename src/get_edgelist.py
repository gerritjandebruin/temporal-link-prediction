import os
import tarfile
import zipfile

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm
import typer

from .logger import logger

app = typer.Typer()


def download(url: str, dst: str, verbose=True):
    """
    @param: url to download file
    @param: dst place to put the file
    @param: if verbose, show tqdm

    Source: https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2
    """
    file_size = int(requests.head(url).headers["Content-Length"])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte, unit='B', unit_scale=True,
        desc=url.split('/')[-1], disable=not verbose)
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()


def extract_tar(tar_file: str, output_path: str) -> None:
    """
    Exctract tar file.

    Args:
      tar_file: File that needs to be extracted.
      path: Location where the contents of the .tar.bz2 will be stored.
    """
    with tarfile.open(tar_file) as tar:
        # I assume that there is only one directory in the tar-archive.
        dir_names = [member.name for member in tar.getmembers()
                     if member.isdir()]
        assert len(dir_names) == 1
        dir_name = dir_names[0]

        tar.extractall(output_path)

    # Move all files from the one directory in the archive to the output_path.
    # On top of that, rename them such that the filename contains only what occurs
    # before the first period.
    # E.g. ./dblp_coauthor/out.dblp_coauthor -> ./out
    with os.scandir(os.path.join(output_path, dir_name)) as it:
        for entry in it:
            os.replace(
                entry.path, os.path.join(output_path, entry.name.split('.')[0])
            )
    os.rmdir(os.path.join(output_path, dir_name))


def from_konect(url: str, *, temp_path: str) -> pd.DataFrame:
    """Download and extract the KONECT dataset. Store extracted files in path. If
    the temporary files are already present in path, the file is not again
    downloaded or extracted. The final edgelist, which is an pd.DataFrame with 
    columns 'source', 'target', 'datetime' is returned.

    Args:
      url: The url pointing to KONECT download file. Usual format: 
        'http://konect.cc/files/download.*.tar.bz2'.
      temp_path: Optional; Store the extracted dataset in this directory.
    """
    logger.debug(f'Start {__file__} with {temp_path=}.')

    # Edgelist is stored in the out.* file contained in the tar archive.
    out_location = os.path.join(temp_path, 'out')
    # Check if extraction took already place.
    if not os.path.isfile(out_location):
        download_location = os.path.join(temp_path, 'download')
        logger.debug('Start download')
        download(url, dst=download_location)

        logger.debug('Start extracting')
        extract_tar(tar_file=download_location, output_path=temp_path)

    # CSV file to pd.DataFrame
    logger.debug('Start reading csv.')
    edgelist = pd.read_csv(
        out_location, delim_whitespace=True, engine='python', comment='%',
        names=['source', 'target', 'weight', 'datetime'])
    edgelist = edgelist[edgelist['datetime'] != 0]

    # Check for signed network
    if -1 in edgelist['weight'].unique():
        print("This is likely a signed network (weight equals -1).\n"
              "Only positive weights will be used.")
        edgelist = edgelist[edgelist['weight'] > 0]

    # Check of both u->v and v->u are present for every edge.
    logger.debug('Check for directionality.')
    edgeset = {
        (u, v) for u, v in edgelist[['source', 'target']].itertuples(index=False)}
    assert np.all(
        [edge in edgeset
         for edge in edgelist[['source', 'target']].itertuples(index=False)])

    # Convert UNIX datetime to datetime object.
    logger.debug('Convert datetime column.')
    edgelist['datetime'] = pd.to_datetime(edgelist['datetime'], unit='s')

    # Check for weights
    logger.debug('Check for weights.')
    if not (edgelist['weight'] == 1).all():
        print('This is a weighted network. However, weights will be discarded.')

    # Drop weight column
    edgelist.drop(columns=['weight'], inplace=True)

    # Store
    return edgelist


def from_aminer(url: str, temp_path: str) -> pd.DataFrame:
    """Download and extract the AMiner dataset. Store extracted files in 
    temp_path. If the temporary files are already present in path, the file is not 
    again downloaded or extracted. The final edgelist, which is an pd.DataFrame with 
    columns 'source', 'target', 'datetime' is returned.

    Args:
      url
      temp_path: Optional; Store the extracted dataset in this directory."""
    # If not yet downloaded, download.
    download_location = os.path.join(temp_path, 'coauthor.zip')
    if not os.path.isfile(download_location):
        download(url, download_location)

    # If not yet extracted, extract.
    logger.debug('Extract')
    extract_location = os.path.join(temp_path, 'coauthor')
    if not os.path.isdir(extract_location):
        with zipfile.ZipFile(download_location, 'r') as zip_ref:
            zip_ref.extractall(temp_path)

    # Read in files.
    logger.debug('Read in')
    with open(os.path.join(extract_location, 'filelist.txt'), 'r') as f:
        filelist = f.read().splitlines()

    # Convert file to pandas dataframe.
    logger.debug('Convert to pd.DataFrame')
    edgelist = pd.concat(
        {
            pd.Timestamp(int(file.split('.')[0]), 1, 1): (
                pd.read_csv(
                    os.path.join(
                        extract_location, file), sep='\t', names=['source', 'target']))
            for file in filelist
        },
        names=['datetime', 'index']
    )
    edgelist.reset_index(level='datetime', inplace=True)
    edgelist.reset_index(drop=True, inplace=True)

    return edgelist


def email_EU(url: str, temp_path: str) -> pd.DataFrame:
    """Download and extract the email EU dataset. Store extracted files in 
    temp_path. If the temporary files are already present in path, the file is not 
    again downloaded or extracted. The final edgelist, which is an pd.DataFrame with 
    columns 'source', 'target', 'datetime' is returned.

    Args:
      url
      temp_path: Optional; Store the extracted dataset in this directory."""
    # If not yet downloaded, download.
    download_location = os.path.join(temp_path, url.split('/')[-1])
    if not os.path.isfile(download_location):
        download(url=url, dst=download_location)

    # Convert file to pandas dataframe.
    logger.debug('Convert to pd.DataFrame')
    edgelist = pd.read_csv(
        download_location,
        delim_whitespace=True,
        index_col=False,
        names=['source', 'target', 'datetime']
    )
    edgelist['datetime'] = pd.to_datetime(edgelist['datetime'], unit='s')

    # Store result
    return edgelist


def reddit(url: str, temp_path: str) -> pd.DataFrame:
    """Download and extract the reddit dataset. Store extracted files in 
    temp_path. If the temporary files are already present in path, the file is not 
    again downloaded or extracted. The final edgelist, which is an pd.DataFrame with 
    columns 'source', 'target', 'datetime' is returned.

    Args:
      url
      temp_path: Optional; Store the extracted dataset in this directory."""
    logger.debug(f'Start construction {temp_path}.')

    # If not yet downloaded, download.
    download_location = os.path.join(temp_path, 'edgelist.tsv')
    if not os.path.isfile(download_location):
        download(url=url, dst=download_location)

    # Convert file to pandas dataframe.
    logger.debug('Convert to pd.DataFrame')
    edgelist = (
        pd.read_csv(
            download_location, sep='\t', index_col=False, parse_dates=['TIMESTAMP'])
        .loc[lambda x: x['LINK_SENTIMENT'] == 1]  # type: ignore
        .rename(columns={'SOURCE_SUBREDDIT': 'source', 'TARGET_SUBREDDIT': 'target',
                         'TIMESTAMP': 'datetime'})
        .loc[:, ['source', 'target', 'datetime']]
    )

    return edgelist


def add_phase(edgelist, split_fraction=2/3,
              t_min=None, t_split=None, t_max=None):
    """Add column to edgelist indicating whether an edge belongs to the mature
    or probe phase.
    
    Arguments:
    edgelist: pd.DataFrame containing source, target and datetime columns
    t_min, optional: Edges before this date are ignored.
    t_split, optional: 
        Edges before this date belong to mature and after to probe.
    t_max, optional: Edges after this date are ignored.
    """
    assert 'phase' not in edgelist.columns
    assert {'source', 'target', 'datetime'}.issubset(set(edgelist.columns))
    # Determine split times
    if t_min is None:
        t_min = edgelist['datetime'].min()  # type: ignore
    else:
        edgelist = edgelist[edgelist['datetime'] > t_min].copy()
    if t_split is None:
        assert split_fraction is not None, (
            "Either split_fraction or t_split should be provided.")
        t_split = edgelist['datetime'].quantile(split_fraction)  # type: ignore
    if t_max is None:
        t_max = edgelist['datetime'].max()  # type: ignore

    # Checks
    assert isinstance(t_min, pd.Timestamp), f"t_min should be pd.Timestamp but is {type(t_min)}"
    assert isinstance(t_split, pd.Timestamp), f"t_split should be pd.Timestamp but is {type(t_split)}"
    assert isinstance(t_max, pd.Timestamp), f"t_max should be pd.Timestamp but is {type(t_max)}"

    # Assign phase column
    edgelist.loc[lambda x: x['datetime'].between(t_min, t_split), 'phase'] = (  # type: ignore
        'mature')
    edgelist.loc[lambda x: x['datetime'].between(t_split, t_max), 'phase'] = (  # type: ignore
        'probe')

    # Checks
    assert 0 < edgelist['phase'].notna().mean() <= 1  # type: ignore

    return edgelist


@app.command()
def single(index_network: int,
           edgelist_path: str,
           split_fraction=2/3,
           t_min=None, t_split=None, t_max=None):
    """
    Download the network and store the result in edgelist_path. 
    Then add a column `phase` which indicates whether an edge belong to 'mature',
    'probe' or None (np.NaN). t_min, t_split can be provided, which should be
    datetime objects used to mark the begin and end of the maturing phase.
    t_split and t_max mark the begin and end of the probing phase. Alternatively,
    split_fraction can be provided, which marks how many percent of edges should
    be in maturing phase and how many in probing phase.
    """
    os.makedirs(os.path.dirname(edgelist_path), exist_ok=True)
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

    # Check if file already exists
    edgelist_path = f'data/{index_network:02}/edgelist.pkl'
    if os.path.isfile(edgelist_path):
        logger.debug(f'{edgelist_path} already exists')
        return

    # Make temp_path if not yet exists
    temp_path = f'temp/{index_network:02}'
    os.makedirs(temp_path, exist_ok=True)

    # Download and extract edgelist
    if index_network in konect_urls:
        edgelist = from_konect(konect_urls[index_network], temp_path=temp_path)
    elif index_network == 7:
        edgelist = from_aminer(
            url='https://lfs.aminer.cn/lab-datasets/dynamicinf/coauthor.zip',
            temp_path=temp_path
        )
    elif index_network == 30:
        edgelist = email_EU(
            url='https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz',
            temp_path=temp_path
        )
    elif index_network == 28:
        edgelist = reddit(
            url='https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv',
            temp_path=temp_path
        )
    elif index_network == 29:
        edgelist = reddit(
            url='https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv',
            temp_path=temp_path
        )
    elif index_network == 5:
        edgelist = pd.read_pickle(edgelist_path)

    else:
        raise Exception(f'Invalid {index_network=}')

    edgelist = add_phase(edgelist, split_fraction, t_min, t_split, t_max)

    edgelist.to_pickle(edgelist_path)


@app.command()
def all():
    """Get all networks and store result in data/%%/edgelist.pkl"""
    for index_network in np.arange(1, 31):
        if index_network != 5:
            single(
                index_network,
                edgelist_path=f'data/{index_network:02}/edgelist.pkl',
                t_min=pd.Timestamp(2001, 1, 10) if index_network == 16 else None
            )


if __name__ == "__main__":
    app()
