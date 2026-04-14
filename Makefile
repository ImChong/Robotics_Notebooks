.PHONY: lint catalog export search ingest log

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

log:
	python3 scripts/append_log.py $(OP) "$(DESC)"
