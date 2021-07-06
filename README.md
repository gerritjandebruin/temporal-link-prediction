This repository provides the code used in a manuscript titled ['Supervised temporal link prediction in large-scale real-world networks'](https://github.com/gerritjandebruin/SNAM2021-paper).

Temporal networks used in this study are downloaded from http://konect.cc/ or http://snap.stanford.edu.

The structure of the directories is as follows:
- code: Contains all code that is not part of tlp package.
- data: Folder where all downloads and intermediate results are stored.
- teexgraph: External dependency to calculate the diameters and shortest path lengths in a network really fast. See .gitmodules.
- tlp: Here most of my code resides. I import this folder as a package in most Jupyter Notebooks.
- cleanup.sh: Clean up Python/ iPython cache folders.
- environment.yml: Can be used to create Python environment with Conda.
- install.sh: Install teexGraph.
- spec-list.txt: Can be used to create exactly the same Python environment.