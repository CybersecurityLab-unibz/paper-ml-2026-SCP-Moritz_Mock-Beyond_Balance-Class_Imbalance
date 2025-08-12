command to build and run a dummy script


pip install . && nlbse_jex --weighted-loss FAMO \
--train ./data/NLBSE25/pharo/pharo_train.csv \
--test ./data/NLBSE25/pharo/pharo_test.csv \
--column-key-text combo \
--column-key-label labels \
--freeze-embeddings True \
--model-path sentence-transformers/paraphrase-MiniLM-L6-v2 \
--batch-size 8 16 32 \
--lr 0.01 0.001 \
--weight-decay 0 0.01 0.001 \
--epoch 5 10

pip install . && nlbse_jex --weighted-loss FAMO \
--train ../data/data/NLBSE25/pharo/pharo_train.csv \
--test ../data/data/NLBSE25/pharo/pharo_test.csv \
--column-key-text combo \
--column-key-label labels \
--model-path microsoft/codebert-base

pip install . && nlbse_jex --weighted-loss EW FAMO test_function.py


python -m unittest discover -s tests

pip install coverage

coverage run -m unittest discover -s tests && coverage report
