.PHONY: lint catalog export search

lint:
	python3 scripts/lint_wiki.py

catalog:
	python3 scripts/generate_page_catalog.py

export:
	python3 scripts/export_minimal.py

search:
	@if [ -z "$(Q)" ]; then \
		echo "用法: make search Q='关键词'"; \
		exit 1; \
	fi
	python3 scripts/search_wiki.py $(Q)
