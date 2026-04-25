.PHONY: lint catalog export export-check search ingest log coverage graph anki slides fetch badge vectors eval-search

lint:
	python3 scripts/eval_search_quality.py
	python3 scripts/lint_wiki.py

catalog:
	python3 scripts/generate_page_catalog.py

export:
	python3 scripts/export_minimal.py

vectors:
	python3 scripts/build_vector_index.py

export-check:
	python3 scripts/check_export_quality.py

eval-search:
	python3 scripts/eval_search_quality.py

search:
	python3 scripts/search_wiki.py $(Q)

ingest:
	python3 scripts/ingest_paper.py $(NAME) --title "$(TITLE)" --desc "$(DESC)" --suggest-updates

log:
	python3 scripts/append_log.py $(OP) "$(DESC)"

sync:
	./scripts/sync_wiki.sh "$(DESC)"

coverage:
	python3 scripts/ingest_coverage.py $(F)

graph:
	python3 scripts/generate_link_graph.py
	python3 scripts/generate_home_stats.py
	cp exports/link-graph.json docs/exports/link-graph.json
	cp exports/graph-stats.json docs/exports/graph-stats.json 2>/dev/null || true
	cp exports/home-stats.json docs/exports/home-stats.json 2>/dev/null || true

anki:
	python3 scripts/export_anki.py

slides:
	python3 scripts/wiki_to_marp.py $(F)

fetch:
	python3 scripts/fetch_to_source.py $(URL) --name $(NAME)

badge:
	python3 scripts/update_badge.py

sync-stats:
	python3 scripts/sync_all_stats.py
