.PHONY: lint catalog export search ingest log coverage graph slides fetch

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

coverage:
	python3 scripts/ingest_coverage.py $(F)

graph:
	python3 scripts/generate_link_graph.py
	cp exports/link-graph.json docs/exports/link-graph.json
	cp exports/graph-stats.json docs/exports/graph-stats.json 2>/dev/null || true

slides:
	python3 scripts/wiki_to_marp.py $(F)

fetch:
	python3 scripts/fetch_to_source.py $(URL) --name $(NAME)
