# 本地命令与 CI 对照

提交前尽量跑齐与 GitHub Actions 相同的检查，避免 PR 反复红。

## 一键对齐 Tests 工作流

与 [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) 中 **Ruff / Mypy / pip-audit / ESLint / Pytest** 顺序一致（不含 Wiki 专项 lint）：

```bash
pip install -r requirements-dev.txt
make ci-test
```

说明：变更 `wiki/` 或导出/索引链时，仍需执行 `make ci-preflight`（见下表）。

## 对照表

| 检查项 | 本地命令 | CI Workflow |
|--------|----------|----------------|
| **Tests 工作流一键** | `make ci-test` | `tests.yml` |
| Wiki lint + 搜索/导出质量 | `make ci-preflight`（或拆：`make lint`、`scripts/ci_preflight.py`） | `search-regression.yml`、`export.yml`（main 自动导出） |
| Ruff（含 import 排序） | `ruff check scripts tests` | `tests.yml` |
| Ruff 格式 | `ruff format scripts tests`（检查：`ruff format --check scripts tests`） | `tests.yml` |
| 静态类型（scripts） | `PYTHONPATH=scripts mypy scripts` | `tests.yml` |
| 单元测试 + 覆盖率阈值（监测模块合计 ≥ **52%**） | `make test` 或 `PYTHONPATH=scripts python3 -m pytest` | `tests.yml` |
| 依赖漏洞审计 | `python3 -m pip_audit -r requirements-dev.txt` | `tests.yml` |
| 圈复杂度参考 | `make complexity`（仅输出，非硬性门禁） | — |
| Wiki lint（轻量依赖） | `python3 scripts/lint_wiki.py` | `lint.yml` |

说明：`make ci-preflight` 会按固定顺序再生目录统计、导出 JSON、图谱统计与 README badge，并执行搜索回归与 wiki lint；变更 wiki/导出链时请以此为准。

轻量 Python 工作流（`lint.yml`、`search-regression.yml`、`export.yml`、`weekly-lint.yml`）共用依赖声明文件 [`requirements-ci-lite.txt`](../requirements-ci-lite.txt)（与 `requirements-dev.txt` 区分），便于 Actions **pip 缓存**命中。

## 依赖安装

```bash
pip install -r requirements-dev.txt
npm ci   # 前端 ESLint（见 package-lock.json）
```

可选：安装 [`pre-commit`](https://pre-commit.com/) 钩子（`pre-commit install`，配置见仓库根目录 `.pre-commit-config.yaml`），提交前自动跑 Ruff。
