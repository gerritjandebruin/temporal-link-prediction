NETWORK_INDICES := $(shell seq -s " " -w 1 30)

# # GET_EDGELIST##: Make edgelist from network with index ##. E.g. make edgelist05 to generate network 5 in data/05/edgelist.pkl
# # get_all_edgelist: Make all edgelists and store them in data/##/edgelist.pkl
# # EDGELIST := $(addprefix get_edgelist,${NETWORK_INDICES})
# .PHONY: get_all_edgelist #GET_EDGELIST ${EDGELIST}
# # GET_EDGELIST: ${EDGELIST}
# # ${EDGELIST}: get_edgelist%: ;	python -m src.get_edgelist single $* data/$*/edgelist.pkl
# get_all_edgelist:
# 	python -m src.get_edgelist all

.PHONY: GET_ALL_INSTANCES
ALL_INSTANCES := $(foreach program, 01 02 03 04 05 06 07 08 09 10 11 12 13 14 16 18 19 20 21 22 23 24 25 28 29 30, data/$(program)/all_instances.npy)
GET_ALL_INSTANCES: ${ALL_INSTANCES}
data/%/all_instances.npy: data/%/edgelist.pkl
	python -m src.get_all_instances single $< $@

.PHONY: SAMPLE
POSITIVES := $(foreach sample, ${NETWORK_INDICES}, data/$(sample)/samples.pkl)
SAMPLE: ${POSITIVES}
data/%/samples.pkl: data/%/edgelist.pkl
	python -m src.sample $< $@ --no-verbose

.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -r {} \+
	find . -type d -name .ipynb_checkpoints -exec rm -r {} \+