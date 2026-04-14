.PHONY: lint catalog export search ingest

lint:
	python3 scripts/lint_wiki.py

catalog:
	python3 scripts/generate_page_catalog.py

export:
	python3 scripts/export_minimal.py

search:
	python3 scripts/search_wiki.py $(Q)

ingest:
	python3 scripts/ingest_paper.py $(NAME) --title "$(TITLE)" --desc "$(DESC)"
