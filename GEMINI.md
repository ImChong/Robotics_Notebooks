# GEMINI.md - Robotics_Notebooks Context

This document serves as the primary instructional context for Gemini CLI when interacting with the `Robotics_Notebooks` project.

## Project Overview
`Robotics_Notebooks` is a structured **Robotics Engineering Knowledge Base** (Wiki), not a simple collection of resources. It focuses on motion control, reinforcement learning (RL), imitation learning (IL), Sim2Real, and humanoid robotics.

### Core Methodology
The project follows the **Karpathy Knowledge Base Methodology**:
- **Three-Layer Architecture:** `sources/` (raw inputs) → `wiki/` (structured compilation) → `schema/` (maintenance rules).
- **Compilation Beats Retrieval:** Wiki pages are distilled knowledge, not just links. Each page must cite its sources.
- **LLM-Driven Maintenance:** LLMs are the primary maintainers (writing, linking, linting), while humans act as curators (directing, auditing).

---

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `wiki/` | **Core Knowledge Layer**. Structured markdown pages (concepts, methods, etc.). |
| `sources/` | **Input Layer**. Raw notes, paper abstracts, blog summaries. |
| `schema/` | **Governance Layer**. Naming conventions, workflows, and page type definitions. |
| `roadmap/` | Curated learning paths and research directions. |
| `tech-map/` | Dependency graphs and technical stack overviews. |
| `scripts/` | Automation tools for linting, searching, indexing, and exporting. |
| `docs/` | Frontend assets (D3.js graph), execution checklists, and website files. |
| `exports/` | Generated data artifacts (JSON/TSV) for the web UI and Anki. |
| `tests/` | Regression tests for search and content integrity. |

---

## Core Operational Workflows (Wiki Ops)

Every interaction must adhere to the workflows defined in `schema/ingest-workflow.md`.

### 1. Ingest (Adding Knowledge)
1. **Source First:** Add raw info to `sources/` (use `make ingest` for papers).
2. **Distill to Wiki:** Create or update a `wiki/` page. **Do not copy-paste**; synthesize and提炼 (distill).
3. **Link Everything:** Add at least 2 cross-references (`related`) to other wiki pages.
4. **Mandatory Section:** Every wiki page **MUST** have a `## 参考来源` (References) section linking back to `sources/`.
5. **Update Index:** Run `make catalog` to refresh `index.md`.
6. **Log Change:** Run `make log OP=ingest DESC="..."`.

### 2. Query (Answering & Synthesis)
1. **Search:** Use `make search Q=...` or semantic search scripts.
2. **Synthesize:** If a query yields a complex synthesis or new insight, write it back to `wiki/queries/` or a new comparison page.
3. **Format:** Query products must start with: `> **Query 产物**：本页由以下问题触发：「问题描述」`.

### 3. Lint (Health Checks)
- Regularly run `make lint` to find orphan pages, broken links, or source coverage gaps.
- Aim for **0 issues**.

---

## Git Commit Standards

All commits must follow the project's historical style and use **Chinese** for descriptions.

### 1. Ingest Commits (Knowledge Addition)
Format: `[YYYY-MM-DD] ingest | <source_path> — <Chinese_description>`
Example: `[2026-04-23] ingest | sources/repos/robot_lab.md — 接入 IsaacLab 扩展框架并更新全站索引`

### 2. Structural & Functional Commits
Format: `<type>(<scope>): <Chinese_description>`
- **type**: feat, fix, chore, docs, refactor, style, test.
- **scope**: Optional (e.g., ux, actions, wiki).
Example: `fix(actions): 修复 CLAW 页面格式缺失主要技术路线的问题`
Example: `chore: 更新主页统计数据与图谱 (172 nodes, 955 edges)`

---

## Key Automation Commands

| Command | Action |
|---------|--------|
| `make lint` | Runs wiki linter and search quality evaluation. |
| `make catalog` | Updates `index.md` based on current `wiki/` files. |
| `make export` | Generates JSON data for the frontend graph and search. |
| `make search Q=...` | Performs a keyword search across the wiki. |
| `make ingest NAME=... TITLE="..."` | Generates a new paper source template. |
| `make graph` | Rebuilds the link graph and stats for the D3 visualization. |
| `make vectors` | Rebuilds the semantic search vector index. |
| `make log OP=... DESC="..."` | Appends an entry to `log.md`. |

---

## Engineering & Content Standards

### 1. Behavioral Principles (from CLAUDE.md)
- **Think First:** State assumptions and tradeoffs before acting.
- **Surgical Changes:** Touch only what is necessary. Maintain local style.
- **Simplicity First:** Avoid over-abstraction. Minimum viable implementation.

### 2. Wiki Writing Style (from AGENTS.md)
- **Concise & Direct:** Use one-sentence definitions at the top.
- **Interconnected:** Prioritize graph structure over hierarchy.
- **Verifiable:** Always provide provenance (links to sources).

### 3. Repository Constraints
- **Do Not** commit or stage changes unless explicitly asked.
- **Do Not** move files between layers without following the `ingest` workflow.
- **Do Not** modify the frontend (`docs/*.js`) unless the request is specifically about UI/UX.

---

## Maintenance Targets (Current Phase)
- **Sources Coverage:** >99%
- **Knowledge Graph:** 217+ nodes, 1255+ edges.
- **Focus Areas:** Contact-rich manipulation, tactile/force feedback, and formalization of communication links.
