FILEPATH ?= "./notebooks/submission.parquet"
MESSAGE ?= "Submission Kaggle Kaggle CLI"

# Install Python dependencies in poetry environment
install:
	poetry install
	poetry run pre-commit install -f

# Upate Python dependencies in poetry environment
update:
	poetry update

# Format src
format:
	poetry run black src
	poetry run isort src
	poetry run black tests
	poetry run isort tests

# Run static code analysis
static-code-analysis:
	poetry run black src --check
	poetry run pylint src --rcfile ../.pylintrc
	poetry run isort src --check-only
	poetry run mypy src --show-error-codes --no-namespace-packages

# Run tests
test:
	poetry run pytest --disable-warnings

# Submit a prediction to Kaggle
submit:
	poetry run kaggle competitions submit -c leap-atmospheric-physics-ai-climsim -f ${FILEPATH} -m ${MESSAGE}
