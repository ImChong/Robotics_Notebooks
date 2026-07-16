# USC PSI Lab — SIMPLE 项目页

> 来源归档（ingest）

- **标题：** SIMPLE · Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation
- **类型：** site（官方项目页）
- **发布方：** USC Physical Superintelligence (PSI) Lab
- **原始链接：** <https://psi-lab.ai/SIMPLE>
- **配套论文：** arXiv:2606.08278 — 归档见 [sources/papers/simple_arxiv_2606_08278.md](../papers/simple_arxiv_2606_08278.md)
- **代码：** <https://github.com/physical-superintelligence-lab/SIMPLE>
- **入库日期：** 2026-07-16
- **一句话说明：** SIMPLE 官方落地页：双仿真器架构说明、60/50/1000+ 规模统计、运动规划与 VR 遥操作数据采集效率表、9 策略 × 6 任务 × L0/L1/L2 benchmark、域随机化/数据 scaling 消融，以及零样本 sim-to-real 并排演示。

## 摘录要点（与论文分工）

- **Overview：** MuJoCo 接触物理 + Isaac Sim 光追渲染；60 whole-body tasks、50 indoor scenes、1000+ objects；内置数据采集与 SOTA policy benchmark。
- **Architecture：** 三阶段管线——(1) MuJoCo 运动规划/遥操作采集；(2) Isaac Sim 离线回放 + 域随机化渲染；(3) Gym 接口下 L0/L1/L2 评测。
- **Scalable Data：** 运动规划（BoDex 抓取 + CuRobo 双臂 + 脚本底座）与 PICO XR VR 遥操作（egocentric 双目 + IK retarget + 全身 tracking policy）。
- **Benchmark：** Ψ₀、GR00T N1.6、π₀.₅、InternVLA、H-RDT、DreamZero、EgoVLA、DP、ACT；每格为 L0/L1/L2 成功次数（满分 10）。
- **Analysis：** 混合 DR 级别训练、遥操作轨迹数 scaling、运动规划 vs 遥操作数据源对比。
- **Transfer：** Pick & Place / Handover 的 sim vs real 成功率与并排视频；策略仅在仿真数据上微调。

## 论文 / 代码状态

- 论文：<https://arxiv.org/abs/2606.08278>
- **已开源：** GitHub `physical-superintelligence-lab/SIMPLE`；评测环境见 Hugging Face `USC-PSI-Lab/psi-data/tree/main/simple-eval`
- 项目页 Footer 链到 arXiv、GitHub 与 HF 数据集

## 对 wiki 的映射

- [SIMPLE 论文实体](../../wiki/entities/paper-loco-manip-161-075-simple.md)
- [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
- [simple_usc_psi 代码归档](../repos/simple_usc_psi.md)
