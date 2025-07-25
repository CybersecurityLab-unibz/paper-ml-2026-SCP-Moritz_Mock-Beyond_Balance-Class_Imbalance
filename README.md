command to build and run a dummy script


pip install . && nlbse_jex --weighted-loss FAMO \
--train ../data/data/NLBSE25/pharo/pharo_train.csv \
--test ../data/data/NLBSE25/pharo/pharo_test.csv \
--column-key-text combo \
--column-key-label labels \
--freeze-embeddings True