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
	$(PY) validate/hbond_reference.py
	$(PY) validate/plip_reference.py
validate: fetch
	cargo test -p proteus-core --release --test validation -- --ignored --nocapture
	cargo test -p proteus-core --release --test hbond_validation -- --ignored --nocapture
	cargo test -p proteus-core --release --test fitness_discrimination -- --ignored --nocapture
	cargo test -p proteus-core --release --test plip_validation -- --ignored --nocapture
