# URDF Files Dataset

> 来源归档

- **标题：** URDF Files Dataset（Daniella1/urdf_files_dataset）
- **类型：** repo / dataset
- **链接：** https://github.com/Daniella1/urdf_files_dataset
- **论文：** Tola & Corke, *Understanding URDF: A Dataset and Analysis*, IEEE RA-L 2024；[arXiv:2308.00514](https://arxiv.org/abs/2308.00514)；DOI [10.1109/LRA.2024.3381482](https://ieeexplore.ieee.org/abstract/document/10478618)
- **作者：** Daniella Tola（奥尔胡斯大学 ECE）、Peter Corke（昆士兰科技大学 QUT Centre for Robotics）
- **许可：** 仓库 **MIT**；各 URDF Bundle 仍受原上来源许可证约束（论文表 XIV：以 BSD-3-Clause 等宽松协议为主）
- **Stars：** 572（2026-08-17 核查）；forks 72
- **默认分支：** `main`（最近推送 **2024-04-06** — 相对 Awesome / robot_descriptions.py 是冻结快照）
- **入库日期：** 2026-08-17
- **一句话说明：** 322 份 URDF Bundle（195 个独特机型）+ 分析脚本：用来研究 URDF 怎么被写、怎么被解析、同机型跨源有何差异，而不是给仿真日常 `import`。
- **开源状态：** **已开源**（MIT；`urdf_files/` + `scripts/`）。无独立项目页。
- **论文摘录：** [understanding_urdf_dataset_arxiv_2308_00514](../papers/understanding_urdf_dataset_arxiv_2308_00514.md)
- **沉淀到 wiki：** [URDF Files Dataset](../../wiki/entities/urdf-files-dataset.md)
- **选型对照：** [机器人描述目录选型](../../wiki/comparisons/robot-description-catalogs.md)

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| GitHub | **已开源**：MIT；README 列出复现论文图表的脚本名 |
| 项目页 | 无 `*.github.io`；论文 HTML/PDF 为学术入口 |
| 数据 | 仓内 `urdf_files/` 即 Bundle 快照；**不是**持续同步的厂商官方仓 |
| 工具 | `scripts/` 可复现表 II–XV 等分析；依赖 ROS `urdfdom` 等解析器做对照 |

## 规模与来源（论文表 II）

| 来源 | Bundle 数 | 变体数 |
|------|-----------|--------|
| ros-industrial | 108 | 1 |
| random | 67 | 39 |
| matlab | 52 | 2 |
| robotics-toolbox | 44 | 15 |
| oems | 35 | 6 |
| drake | 16 | 12 |
| **合计** | **322** | **75** |

独特机型 **195**；多余条目是跨源重复定义或碰撞/接触变体（如 iiwa 多种 collision、Atlas convex hull vs minimal contact）。

## 论文级发现（写入 wiki 时用）

- **95%** Bundle 经 xacro 生成。
- 官方 ROS `urdfdom` 3.1.0：**11/322** 解析失败（最常见 XML parsing failed）。
- 网格：视觉 STL 最多，其次 COLLADA / OBJ；视觉网格 Bundle 341 vs 碰撞 278（可无碰撞体）。
- **60** 个机型跨源多重定义（130 Bundle，均 2.2 源/机）；同名机器人文件不必相同。
- 多解析器对照表明「Unified」并不统一：各 parser 对同一文件的接受度不同。
- ros-industrial 约占 34%，分析文件夹结构 / xacro / 网格类型时有来源偏差。

## 对 wiki 的映射

- [URDF Files Dataset](../../wiki/entities/urdf-files-dataset.md)
- [URDF](../../wiki/concepts/urdf-robot-description.md)
- [机器人描述目录选型](../../wiki/comparisons/robot-description-catalogs.md)
- [URDD](../../wiki/entities/paper-urdd-universal-robot-description-directory.md) — 派生层 vs 本集「原始 URDF 语料」
