.PHONY: all
all:
	./run.sh

.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -r {} \+
	find . -type d -name .ipynb_checkpoints -exec rm -r {} \+