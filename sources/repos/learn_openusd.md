# LearnOpenUSD

> 来源归档

- **标题：** NVIDIA-Omniverse/LearnOpenUSD
- **类型：** repo
- **来源：** NVIDIA Omniverse
- **链接：** https://github.com/NVIDIA-Omniverse/LearnOpenUSD
- **渲染站：** https://docs.nvidia.com/learn-openusd/latest/index.html
- **入库日期：** 2026-08-30
- **通读日期：** 2026-11-28
- **核对范围：** 通读 `NVIDIA-Omniverse/LearnOpenUSD` 仓库 `README.md`、`docs/index.md`（课程总览、认证入口与 9 大章节 `toctree`）、官方文档渲染首页目录与架构说明（`docs.nvidia.com/learn-openusd/latest/index.html`）以及练习样例脚本 `docs/exercise_content/asset_structure/exercise_01/make_entry_point_full.py`（45 行、`pxr.Usd` + `UsdGeom.Xform` + `Usd.NamespaceEditor`）。
- **通读：** 已完成
- **一句话说明：** Learn OpenUSD 官方开源课程仓（Apache-2.0），以 **uv + Git LFS + Sphinx + MyST-NB** 构建，提供从 Stage/Prim 基础到 Asset Structure、Composition Arcs（LIVERPS）、Data Exchange 与 Instancing 的 Python/Notebook 实操课纲。
- **代码：** https://github.com/NVIDIA-Omniverse/LearnOpenUSD（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-learn-openusd.md`](../../wiki/entities/nvidia-learn-openusd.md)

---

## 一手构建与课纲源码核对（2026-11-28）

1. **构建流水线（`README.md`）**：
   - 使用 `uv` 管理 Python 依赖，使用 **Git LFS** 拉取图像、视频与 USD 教学资产（`git lfs install && git lfs pull`）。
   - 执行 `uv run sphinx-build -M html docs/ docs/_build/` 时按 5 步执行：**MyST-NB 运行 Python 代码单元** → Sphinx 解析指令与交叉引用 → 自定义扩展复制资产并打包练习 ZIP → 提取交互式词汇表图谱 → 生成带自定义主题的 HTML（支持 `uv run python -m http.server 8000 -d docs/_build/html/` 预览或 `uv run launch_notebooks` 运行 Notebook）。
2. **课程主干（`docs/index.md` & Rendered TOC）**：
   - 认证对接：**OpenUSD Development Certification** 与 **NCA OpenUSD Foundations and Workflows**（提供 `nca-ousd-study-guide`、`openusd-cheatsheet`、交互式 Glossary）。
   - 核心模块依次为：`What Is OpenUSD` → `Setting the Stage`（Stage/Prims/Properties/TimeCodes/Paths/FileFormats/Modules/Metadata）→ `Scene Description Blueprints`（Schemas/Scope/Xform/XformCommonAPI/Lights）→ `Composition Basics`（Layers/Specifiers/Referencing/Default Prim/Variant Sets）→ `Beyond the Basics`（Primvars/Value Resolution/Value Clips/Spline Animation/Model Kinds/Hydra/Units）→ `Creating Composition Arcs`（Sublayers/References & Payloads/Encapsulation/Inherits & Specializes/LIVERPS 优先级追踪）→ `Asset Structure Principles and Content Aggregation`（Asset Interface/Workstreams/Parameterization/Reference-Payload Pattern/Components-Assemblies-Groups 模型层级）→ `Developing Data Exchange Pipelines`（Converter 解剖/几何材质提取/Asset Validation/Prim 层级转换）→ `Asset Modularity and Instancing`（Scenegraph Instancing/Nested Instancing/Deinstancing 与 Broadcasted Refinement/Point Instancing）。
3. **练习脚本示例核对（`make_entry_point_full.py`）**：
   - 以 `from pxr import Usd, UsdGeom` 打开 `export.usd`，定义 `/World` `Xform` 并调用 `stage.SetDefaultPrim(world_prim)`，再用 `Usd.NamespaceEditor(stage)` 调用 `ReparentPrim` + `ApplyEdits()` 把既有根 prim 重挂到统一入口 prim 下以维持材质绑定等关系路径。
4. **证据边界**：课纲仓本身仅是教学/文档构建项目与练习代码，**不包含** Pixar OpenUSD C++/Python 内核源码或 Omniverse Kit 运行时（仿真执行侧见 OpenUSD `UsdPhysics`、NVIDIA Omniverse Libraries 与 Isaac Sim）。

---

## 核心定位

- **内容形态：** RST/Sphinx 教程 + 练习脚本；非 OpenUSD 运行时本身
- **受众：** 3D 管线开发者、Omniverse/Isaac 资产作者、OpenUSD 认证考生
- **贡献：** 课纲在 GitHub 开源，与 [OpenUSD.org](https://openusd.org/) / NVIDIA OpenUSD Docs 并列延伸阅读

---

## 关联档案

- 课程归档：[`nvidia_learn_openusd.md`](../courses/nvidia_learn_openusd.md)
- Physical AI 门户：[`nvidia-physical-ai-learning.md`](../sites/nvidia-physical-ai-learning.md)
