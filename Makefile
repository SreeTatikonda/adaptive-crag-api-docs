.PHONY: ingest index api ui benchmark test clean install

install:
	pip install -e ".[dev]"

ingest:
	python scripts/ingest_docs.py

index:
	python scripts/build_indices.py

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run src/ui/streamlit_app.py

benchmark:
	python scripts/run_benchmark.py

export:
	python scripts/export_metrics.py

test:
	pytest tests/ -v

clean:
	rm -rf data/raw/* data/processed/chroma data/processed/bm25_index.pkl data/processed/traces.jsonl
