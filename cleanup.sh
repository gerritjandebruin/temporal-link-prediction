# Clean up recursively Python/iPython cached folders in current working directory.
find . -type d -name __pycache__ -exec rm -r {} \+
find . -type d -name .ipynb_checkpoints -exec rm -r {} \+