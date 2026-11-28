# robot_descriptions.py 官方文档与核心加载源码通读

> 来源归档（原文通读 · 支撑 `wiki/entities/robot-descriptions-py.md` 与 `wiki/comparisons/robot-description-catalogs.md`）

- **标题：** robot-descriptions/robot_descriptions.py（README、pyproject.toml 与核心加载源码）
- **作者 / 来源：** Stéphane Caron & robot-descriptions contributors（INRIA 谱系）
- **类型：** repo README & Python source code
- **原文：** <https://github.com/robot-descriptions/robot_descriptions.py>（v3.4.0，2026-11-27 发布）
- **入库日期：** 2026-11-28
- **一句话说明：** 核读 `robot_descriptions.py` v3.4.0 的 README、清单索引（229 条机器人描述）与缓存/Xacro/Loader/CLI 源码，确认按名按需浅克隆缓存、多仿真/动力学后端适配及主包 Apache-2.0 与上游资产逐条许可的边界。
- **沉淀到 wiki：** 是 → [`wiki/entities/robot-descriptions-py.md`](../../wiki/entities/robot-descriptions-py.md)、[`wiki/comparisons/robot-description-catalogs.md`](../../wiki/comparisons/robot-description-catalogs.md)
- **通读：** 已完成
- **核对范围：** 通读官方 `README.md`、`pyproject.toml`、`robot_descriptions/__init__.py`、`robot_descriptions/__main__.py`、`robot_descriptions/_cache.py`、`robot_descriptions/_descriptions.py`（统计定义条目）、`robot_descriptions/a1_description.py`、`robot_descriptions/loaders/mujoco.py`、`robot_descriptions/loaders/pinocchio.py`、`robot_descriptions/loaders/yourdfpy.py` 与 `examples/load_in_mujoco.py`（`main` 分支，v3.4.0 期间，2026-11-28 抓取）。

## 核心摘录（均出自仓库原文与源码）

1. **版本与规模核对：** GitHub 最新 Release 为 **`v3.4.0`**（2026-11-27），`pyproject.toml` 声明 **Python >= 3.11**、`flit_core` 构建、核心依赖含 **GitPython >= 3.2.1**；`robot_descriptions/_descriptions.py` 共注册 **229 个** `Description(...)` 条目（README 概览仍写作 190+）。
2. **模块与延迟缓存机制：** 每个具体机型子模块（如 `robot_descriptions/a1_description.py`）在首次 `import` 时即执行 `_cache.clone_to_cache(<repo_key>, commit=...)`，将对应上游仓浅拉取（`depth=1`）到 `$ROBOT_DESCRIPTIONS_CACHE`（默认 `~/.cache/robot_descriptions`），并保留与 ROS `package://` URI 兼容的末端包目录名，随后导出 `REPOSITORY_PATH`、`PACKAGE_PATH`、`URDF_PATH` / `MJCF_PATH` 等常量；若声明 `XACRO_PATH`，则由 `_xacro.get_urdf_path` 透明展开并缓存 URDF。
3. **6 套 Loader 与 CLI 入口：** 提供 `robot_descriptions.loaders.{idyntree, mujoco, pinocchio, pybullet, robomeshcat, yourdfpy}`，核心函数 `load_robot_description(description_name, ...)` 内部先设置可选 `ROBOT_DESCRIPTION_COMMIT` 环境变量再 `import_module(f"robot_descriptions.{description_name}")`，根据 `URDF_PATH` / `XACRO_PATH` / `MJCF_PATH` 分派至对应后端构建函数（如 `mujoco.MjModel.from_xml_path`、`pin.RobotWrapper.BuildFromURDF` / `BuildFromMJCF`、`yourdfpy.URDF.load`）；CLI 入口 `__main__.py` 支持 `list`、`pull` 与 `show_in_{meshcat,mujoco,pybullet,yourdfpy}`。
4. **许可边界：** Python 包本身为 **Apache-2.0**，但各个模型由各自上游仓按 `REPOSITORIES` 条目克隆，具体网格与描述文件受上游 README 列出的独立许可证约束（含 Apache-2.0、BSD-3-Clause、MIT、CC-BY-NC 及厂商图形文档条款等）。

## 对 wiki 的映射

- [`wiki/entities/robot-descriptions-py.md`](../../wiki/entities/robot-descriptions-py.md)
- [`wiki/comparisons/robot-description-catalogs.md`](../../wiki/comparisons/robot-description-catalogs.md)
