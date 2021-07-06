TEMP_PATH='temp'

mkdir -p $TEMP_PATH
git clone git@github.com:franktakes/teexgraph.git $TEMP_PATH
git --git-dir $TEMP_PATH checkout -b old-state 0c4ebef4ee938aa842bf40d1aec8a66d95fd8a82
(cd $TEMP_PATH && make listener)

python -m src.get_edgelist all
python -m src.rewire all
python -m src.get_samples all
python -m src.get_features all
python -m src.get_performance all