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


setup hook 
pip install black

cd .git/hook
touch pre-commit

paste code below

```commandline
#!/bin/bash

echo "Running black to auto-format staged Python files..."

# Get list of staged Python files
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -z "$FILES" ]; then
  echo "No Python files to format."
  exit 0
fi

# Format each staged Python file and re-add it to staging
for file in $FILES; do
  black "$file"
  git add "$file"
done

echo "Formatting complete. Proceeding with commit."
exit 0
```

chmod -x pre-commit