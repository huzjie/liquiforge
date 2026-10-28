.PHONY: install test bench doctor serve docker clean

install:
	pip install -e .

test:
	python -m unittest discover -s tests -t .

bench:
	python scripts/run_bench.py

doctor:
	python -m liquiforge.cli doctor

serve:
	python -m liquiforge.cli serve

docker:
	docker build -t liquiforge:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .push_cache.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
