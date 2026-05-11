.PHONY: lint test ci-test install-hooks format lint-py lint-js typecheck complexity audit-dev catalog export export-check search ingest log coverage graph anki slides fetch badge vectors eval-search ci-preflight ci-check

# 与 .github/workflows/tests.yml 步骤顺序一致（不含 Wiki lint）
ci-test:
	ruff check scripts tests
	ruff format --check scripts tests
	PYTHONPATH=scripts mypy scripts
	python3 -m pip_audit -r requirements-dev.txt
	npm ci
	npm run lint:js
	PYTHONPATH=scripts python3 -m pytest

lint:
	python3 scripts/eval_search_quality.py
	python3 scripts/lint_wiki.py

# 与 CI tests job 对齐：pytest（含覆盖率阈值）、ruff、mypy、pip-audit 见下方目标
test:
	PYTHONPATH=scripts python3 -m pytest

install-hooks:
	pre-commit install

format:
	ruff format scripts tests

lint-py:
	ruff check scripts tests
	ruff format --check scripts tests

lint-js:
	npx --yes eslint docs/main.js

typecheck:
	PYTHONPATH=scripts mypy scripts

complexity:
	radon cc scripts -a -nc | tail -30

audit-dev:
	python3 -m pip_audit -r requirements-dev.txt

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
	python3 scripts/graph_exports_sync.py

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

ci-preflight:
	python3 scripts/ci_preflight.py

ci-check:
	python3 scripts/ci_preflight.py --check-generated-clean
