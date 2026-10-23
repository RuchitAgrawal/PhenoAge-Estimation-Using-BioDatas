.PHONY: help setup eda train evaluate pipeline app test clean

help:
	@echo "PhenoAge ML Research Pipeline Tasks:"
	@echo "  make setup      - Install Python dependencies"
	@echo "  make pipeline   - Execute end-to-end Python ML reproduction pipeline"
	@echo "  make eda        - Generate cohort exploratory figures and tables"
	@echo "  make app        - Launch interactive clinical Streamlit web application"
	@echo "  make test       - Run unit tests with pytest"
	@echo "  make clean      - Clean bytecode and temporary cache files"

setup:
	pip install -r requirements/requirements.txt

pipeline:
	python run_pipeline.py

eda:
	python -c "from src.data_loader import load_data; from src.eda import run_eda; run_eda(load_data())"

app:
	streamlit run app.py

test:
	pytest tests/ -v --tb=short

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
