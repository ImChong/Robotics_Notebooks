# SOMA-X 官方文档站

> 来源归档（ingest）

- **标题：** SOMA-X Documentation
- **类型：** site（API 文档 + 项目文档合一）
- **链接：** https://nvlabs.github.io/SOMA-X/stable/
- **关联仓库：** https://github.com/NVlabs/SOMA-X
- **入库日期：** 2026-06-17
- **一句话说明：** API-first 文档站，以 `soma` Python 库为中心，覆盖 `SOMALayer`、`PoseInversion`、`soma.io`（NPZ/USD）与 `soma.geometry`（FK、LBS、skeleton fitting、Warp kernels）；与仓库 `docs/` 同步纳入导航。
- **沉淀到 wiki：** [SOMA-X](../../wiki/entities/soma-x.md)

## 摘录要点

- **受众定位：** 开发者优先的 API 参考，而非营销型项目页。
- **核心模块：**
  - `SOMALayer` — 全身体参数化前向（身份 + 姿态 + 可选 scale）
  - `PoseInversion` — 从其他模型姿态参数拟合到 SOMA（analytical / autograd FK）
  - `soma.io` — NPZ 与 USD 工作流辅助
  - `soma.geometry` — FK、LBS、skeleton fitting、Warp 内核
- **本地构建：** `uv pip install -e ".[docs]"`；`SOMA_DOCS_AUDIENCE=public DOC_VERSION=0.2 sphinx-build -b html docs docs/_build/html`。
- **版本：** 站点支持版本选择（stable）；与 PyPI `py-soma-x` 发布对齐。

## 对 wiki 的映射

- [SOMA-X](../../wiki/entities/soma-x.md) — 「核心 API / 工程要点」与安装说明互链
