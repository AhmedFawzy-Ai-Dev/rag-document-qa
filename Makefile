.PHONY: install ingest ask ui test lint clean
install:
	pip install -e ".[dev]"
ingest:
	python -m docqa.ingest
ask:
	python -m docqa.ask "What is data leakage and how do you detect it?" --show-sources
ui:
	python app.py
test:
	pytest -q
lint:
	ruff check src tests app.py
clean:
	python -c "import pathlib; pathlib.Path('data/index.joblib').unlink(missing_ok=True)"
