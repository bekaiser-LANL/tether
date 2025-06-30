.PHONY: install
install:
	python3 -m pip install .

.PHONY: dev_install
dev_install:
	python3 -m pip install '.[dev,test]'

.PHONY: lint
lint:
	python3 -m pylint example_project/

.PHONY: format
format:
	python3 -m isort example_project tests/
	python3 -m black example_project/ tests/

.PHONY: test
test:
	python3 -m pytest tests/
