.PHONY: install
install:
	python3 -m pip install .

.PHONY: dev_install
dev_install:
	python3 -m pip install -e '.[dev,test]'

.PHONY: lint
lint:
	python3 -m pylint tether/

.PHONY: format
format:
	python3 -m isort tether tests/
	python3 -m black tether/ tests/

.PHONY: test
test:
	python3 -m pytest tests/
