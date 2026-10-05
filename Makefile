.PHONY: ingest test

PYTHON ?= python

ingest:
	$(PYTHON) scripts/ingest_docs.py

test:
	$(PYTHON) -m pytest
