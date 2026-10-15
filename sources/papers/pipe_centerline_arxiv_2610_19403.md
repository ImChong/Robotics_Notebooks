# Topology-Preserving Multi-Usage Pipe Centerline Refinement under Permanent Tubular Occupancy（arXiv:2610.19403）

> 来源归档（ingest）

- **标题：** Topology-Preserving Multi-Usage Pipe Centerline Refinement under Permanent Tubular Occupancy
- **arXiv：** <https://arxiv.org/abs/2610.19403>（PDF：<https://arxiv.org/pdf/2610.19403>；DOI：10.48550/arXiv.2610.19403）
- **提交日期：** 2026-10-14（v1，cs.RO；6,924 KB）
- **作者：** Deng Quan（提交记录中署名为 Quan Deng），单人署名
- **机构：** 论文页未列出。
- **文章所附代码（未确认关联）：** <https://github.com/bladesaber/MAPF_Pipeline>
- **入库日期：** 2026-10-15
- **一句话说明：** 在"离散多组管路布线已给出可行粗布局与拓扑"之后，对永久占据管路的中心线做拓扑保持的几何精修（主干共享 / 分支汇合参数化、软几何代价、硬间隙验收）；压降物理优化留待后续 PDE 阶段。

## 开源状态（步骤 2.5，2026-10-15）

- **论文页核查：** 页面只链接 PDF、TeX 源码、DOI 与通用第三方工具，未列出任何代码仓库。
- **文章所附仓库核查：** `bladesaber/MAPF_Pipeline` 公开可见（119 commits，0 stars），但仓库无描述、无 LICENSE 可见；README 列出的依赖为 Eigen3、Netgen、FEniCSx、Gmsh（网格与有限元），未提及管路中心线或本论文；目录为 `include/ src/ scripts_py/ reference/ media/`，名称含 "MAPF"（多智能体路径规划），与论文页自述的问题定义不符。
- **结论：** **未开源**（论文未声明代码）。文章所附仓库与本论文的对应关系**未核实**，不作为论文官方代码引用。

## 摘录 1：问题与主张（Abstract / 正文，转述）

- **痛点：** 离散多组管路布线（网格或粗图上）得到的可行中心线常含大量轴向直角转折；对增材制造的液压歧管等永久占据的流体管路，这些转折损害几何公平性与流动质量。
- **设定：** 起点为给定的路由拓扑与可行粗布局；对象为带声明终端用途、有限管半径、跨组静态占据的分支歧管。与经典机器人路径平滑（对单条临时开放轨迹做 fairing）不同。
- **问题形式化：** 指出 "多用途交汇冲突（multi-usage junction conflict）"，即单一共享分支点无法同时满足各终端用途。
- **方法：** 混合参数化（主干共享 trunk-share / 分支汇合 junction-split），软几何代价，硬间隙验收（hard clearance acceptance）。压降的物理优化推迟到后续 PDE 阶段。

## 摘录 2：实验（转述）

- 设置：多组套件，8 组 × 8 终端，管半径 r = 2，胶囊体（capsule）净空判定。
- 报告指标：完全硬接受成功率（full hard-accept success）、κr ≲ 1、软目标相对下降约 95%。
- 数据集：未命名公开数据集；页面未描述。

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-pipe-centerline-refinement.md`](../../wiki/entities/paper-pipe-centerline-refinement.md)。
- 文章给出的 MAPF_Pipeline 仓库不作为本论文代码；若作者后续公布正式实现，再更新 `code:` 字段。
