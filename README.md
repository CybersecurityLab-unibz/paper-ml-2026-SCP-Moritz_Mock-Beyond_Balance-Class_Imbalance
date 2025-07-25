command to build and run a dummy script


<<<<<<< HEAD
pip install . && nlbse_jex --weighted-loss FAMO \
--train ../data/data/NLBSE25/pharo/pharo_train.csv \
--test ../data/data/NLBSE25/pharo/pharo_test.csv \
--column-key-text combo \
--column-key-label labels \
--freeze-embeddings True
=======
pip install . && nlbse_jex --weighted-loss EW FAMO test_function.py


python -m unittest discover -s tests

pip install coverage

coverage run -m unittest discover -s tests && coverage report
>>>>>>> main
