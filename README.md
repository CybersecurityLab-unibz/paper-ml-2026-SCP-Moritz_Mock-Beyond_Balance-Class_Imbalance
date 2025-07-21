command to build and run a dummy script


pip install . && nlbse_jex --weighted-loss EW FAMO test_function.py


python -m unittest discover -s tests

pip install coverage

coverage run -m unittest discover -s tests && coverage report
