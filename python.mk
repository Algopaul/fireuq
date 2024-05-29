.venv:
	python3.11 -m venv .venv
	.venv/bin/pip install -e .

install: pyproject.toml | .venv
	.venv/bin/pip install -e .

PYTHON = .venv/bin/python
