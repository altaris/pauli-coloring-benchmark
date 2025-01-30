SRC_PATH 	= pcb
LIB_PATH 	= $(SRC_PATH)/lib
DOCS_PATH 	= docs

PDOC		= pdoc -d google --math
PYTHON		= python3.12

RUFF_EXCL   = --exclude '*.ipynb' --exclude 'old/' --exclude 'playground.py' --exclude test.py

.ONESHELL:

all: format typecheck lint

dll:
	gcc $(LIB_PATH)/coloring.c -shared -o $(LIB_PATH)/coloring.so -O3

dll-debug:
	gcc $(LIB_PATH)/coloring.c -shared -o $(LIB_PATH)/coloring.so -fPIC -g -O0

.PHONY: docs
docs:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	PDOC_ALLOW_EXEC=1 uv run $(PDOC) --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	PDOC_ALLOW_EXEC=1 uv run $(PDOC) -p 8081 -n $(SRC_PATH)

.PHONY: format
format:
	uvx ruff check --select I --fix $(RUFF_EXCL)
	uvx ruff format $(RUFF_EXCL)

.PHONY: lint
lint:
	uvx ruff check --fix $(RUFF_EXCL)
	clang-format --style=Microsoft -i $(LIB_PATH)/*.c

.PHONY: typecheck
typecheck:
	uv run mypy -p $(SRC_PATH)