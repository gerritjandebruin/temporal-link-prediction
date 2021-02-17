This repository provides the experimental work I perform for a paper that I am currently writing.
It is not very structured, as I usually write code in a hurry for the next deadline.
I am sorry. :)

The paper is about temporal link prediction.
The current scope is to apply explainable, scalable, temporal features to a set of 31 temporal networks.
Most networks are imported from http://konect.cc/ or http://snap.stanford.edu.

For the Code Refinery Workshop, please have a look at [this folder](./tlp), as it contains pure Python code (no notebooks) and this code is used all-over the repository.

Disclaimer:
- I work on my own in this project, so no real collaboration.

Nevertheless, the structure of the directories is as follows:
- data: Folder where all downloads and intermediate results are stored.
- features: Folder where the analysis of the performance of some features are stored.
- notebooks: Folder containing all the checks whether a network was imported rightly. I used these notebooks to run code as well.
- scores: WIP; Should contain the ROC-AUC scores obtained after employing ML.
- teexgraph: External dependency to calculate the diameters and shortest path lengths in a network really fast. See .gitmodules.
- tlp: Here most of my code lives. I import this folder as a package in most Jupyter Notebooks.
- calc-1.py: Ignore
- dependencies.ipynb: Instructions to compile ./teexgraph.
- sp.py: Ignore
- spec-list.txt: Can be used to create exactly the same Python environment.
- stats.ipynb: Contains several statistics of all networks used in this study.