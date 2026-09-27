VENV := validate/.venv
PY := $(VENV)/bin/python

$(VENV):
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -r validate/requirements.txt

.PHONY: fetch reference geometry-reference validate
fetch: $(VENV)
	$(PY) validate/fetch.py
reference: fetch
	$(PY) validate/reference.py
	$(PY) validate/hbond_reference.py
	$(PY) validate/plip_reference.py
# cctbx covalent geometry, Cβ, ω and rotamers; needs PROTEUS_CHEM_DATA from
# validate/fetch_chem_data.sh. Kept out of `reference` because of that extra download.
geometry-reference: fetch
	$(PY) validate/geometry_reference.py
	gzip -9f validate/reference/geometry/*.json
validate: fetch
	cargo test -p proteus-core --release --test validation -- --ignored --nocapture
	cargo test -p proteus-core --release --test hbond_validation -- --ignored --nocapture
	cargo test -p proteus-core --release --test fitness_discrimination -- --ignored --nocapture
	cargo test -p proteus-core --release --test plip_validation -- --ignored --nocapture
	cargo test -p proteus-core --release --test geometry_validation -- --ignored --nocapture
	cargo test -p proteus-core --release --test rotamer_validation -- --ignored --nocapture
