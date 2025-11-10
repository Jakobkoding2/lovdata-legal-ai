PYTHON=python

.PHONY: data chunks embed api eval update

data:
	$(PYTHON) scripts/data_pipeline.py

chunks:
	$(PYTHON) scripts/chunk_pipeline.py

embed:
	$(PYTHON) scripts/embed_pipeline.py

api:
	$(PYTHON) api/api_server.py

update:
	$(PYTHON) scripts/update_from_api.py

eval:
	$(PYTHON) scripts/eval_rag.py --limit 10
