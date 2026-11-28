# Understanding URDF（arXiv:2308.00514）与四大描述目录一线仓库通读

> 来源归档（原文通读 · 支撑 `wiki/comparisons/robot-description-catalogs.md`）

- **标题：** Understanding URDF: A Dataset and Analysis（arXiv:2308.00514v1）及 Awesome / fiveages-sim / urdf_files_dataset / MuJoCo Menagerie 一线仓库 README
- **作者 / 来源：** Daniella Tola, Peter Corke（2023-08）及对应开源社区/厂商维护仓
- **类型：** paper + official repo READMEs
- **原文：**
  - 论文 PDF：<https://arxiv.org/pdf/2308.00514>（arXiv:2308.00514v1，9 页）
  - 数据集仓：<https://github.com/Daniella1/urdf_files_dataset>
  - Awesome 列表：<https://github.com/robot-descriptions/awesome-robot-descriptions>
  - fiveages-sim 仓：<https://github.com/fiveages-sim/robot_descriptions>
  - MuJoCo Menagerie：<https://github.com/google-deepmind/mujoco_menagerie>
- **入库日期：** 2026-11-28
- **一句话说明：** 通读 Tola & Corke (2023) 对 322 份 URDF（195 个唯一机器人模型）的实证统计与 6 种 URDF 解析器对比，并核对当前四大机器人描述目录仓库 README 的结构、分发形式与许可边界。
- **沉淀到 wiki：** 是 → [`wiki/comparisons/robot-description-catalogs.md`](../../wiki/comparisons/robot-description-catalogs.md)
- **通读：** 已完成
- **核对范围：** 通读 `arXiv:2308.00514v1` 全文 9 页（I–VI 节、Tables I–XVI）以及上述四份 GitHub 仓库官方 `README.md` 全文（2026-11-28 抓取）。

## 核心摘录（均出自原文与仓库 README）

1. **URDF 生态实证（Tola & Corke 2023）：**
   - **规模与来源分布：** 收集 6 类来源共 **322 个 URDF 文件**（其中 **195 个** 唯一模型，其余为多源复现或变体）：`ros-industrial` (108, 34%)、`random` (67)、`matlab` (52)、`robotics-toolbox` (44)、`oems` (35)、`drake` (16)。
   - **厂商供给率与检索可见度：** **16/30** 原厂至少为一个机型提供 URDF Bundle（工业向 8/20，非营利/科研向 6/7）；仅 **4/16** 在自家官网检索能查到 URDF 信息。
   - **构建与语法问题：** **95%（255/270）** 由 `xacro` 生成；使用官方 ROS 解析器 `urdfdom 3.1.0` 时仍有 **11/322（3.4%）** 报错失败（主要为 XML 解析失败、缺 joint limit effort/velocity、缺 link 或父子连杆未定义）；6 种解析器对比表（Table XVI）显示通过率差异巨大（`urdfpy` 仅 7/322、`yourdfpy` 220/322、`pybullet` 252/322、`robotics-toolbox` 306/322、`MATLAB` 309/322、`ROS urdfdom` 311/322），证明不同工具对 URDF 规则解释并不统一。
   - **网格与许可：** 视觉网格以 STL 为绝对主流；数据集中最常见的开源许可为 BSD-3-Clause，且几乎不写作者联系元数据（仅有 13/16 个文件出现 author / `@`）。
2. **一线目录仓结构差异：**
   - `awesome-robot-descriptions`（MIT/CC0 清单索引）按机械臂/双足/双臂/无人机/教育/末端/人形/移动操作/四足/轮式分组，逐行标明 Formats、License 与 Visuals / Inertias / Collisions 三栏勾选；
   - `fiveages-sim/robot_descriptions` 为 ROS 2 包集合，重点收录国内轮式人形/双足/机械臂（含 DexForce W1、Agibot G1/A2、Galaxea R1/A1、Unitree G1、EngineAI、Piper 等），大量使用 Blender 重绘外观，并以 Git Submodule 管理部分机型（含私有/未公开 submodule 如 Agibot G2）；
   - `google-deepmind/mujoco_menagerie` 为 MJCF 权威资产库，每目录分离独立 `<model>.xml`（无多余世界实体）与 `scene.xml`（含地面/光照，部分提供 `*_mjx.xml`），提供 `pip install mujoco-menagerie`；
   - `urdf_files_dataset` 按论文冻结分类打包 322 份 URDF Bundle 与复现分析脚本，用于研究与解析器基准而非实时新机型更新源。

## 对 wiki 的映射

- [`wiki/comparisons/robot-description-catalogs.md`](../../wiki/comparisons/robot-description-catalogs.md)
