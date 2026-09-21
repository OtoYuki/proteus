VENV := validate/.venv
PY := $(VENV)/bin/python

$(VENV):
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -r validate/requirements.txt

.PHONY: fetch reference validate
fetch: $(VENV)
	$(PY) validate/fetch.py
reference: fetch
	$(PY) validate/reference.py
validate: fetch
	cargo test -p proteus-core --release --test validation -- --ignored --nocapture
