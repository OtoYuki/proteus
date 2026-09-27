VENV := validate/.venv
PY := $(VENV)/bin/python

$(VENV):
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -r validate/requirements.txt

.PHONY: fetch reference geometry-reference validate validate-binders validate-nipah
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
# Binder triage against 3 669 wet-lab-tested designs (Overath et al. 2025). Downloads ~2 GB into
# ~/.cache/proteus-validate/binders once; not part of CI. Writes validate/binders/last_run.md.
BINDERS ?= $(HOME)/.cache/proteus-validate/binders
validate-binders:
	PROTEUS_BINDERS=$(BINDERS) validate/binders/fetch.sh
	cargo build --release -p proteus-cli
	target/release/proteus analyze $(BINDERS)/af3/AF3_outputs --interface A --json > $(BINDERS)/af3.jsonl
	python3 validate/binders/compare.py $(BINDERS)/af3.jsonl $(BINDERS)/final_dataset.csv --report validate/binders/last_run.md

# Binder triage against the Adaptyv Nipah competition (1 200 designs, one lab, Boltz-2 models with
# full PAE). Downloads ~11 GB into ~/.cache/proteus-validate/nipah once; not part of CI. Writes
# validate/nipah/last_run.md.
NIPAH ?= $(HOME)/.cache/proteus-validate/nipah
validate-nipah:
	PROTEUS_NIPAH=$(NIPAH) validate/nipah/fetch.sh
	cargo build --release -p proteus-cli
	target/release/proteus analyze $(NIPAH)/models --interface B --json > $(NIPAH)/boltz2.jsonl
	python3 validate/nipah/compare.py $(NIPAH)/boltz2.jsonl $(NIPAH)/collection.csv --report validate/nipah/last_run.md
