# SIMPLE（USC PSI Lab）

> 来源归档（ingest）

- **标题：** SIMPLE — Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation
- **类型：** repo
- **链接：** <https://github.com/physical-superintelligence-lab/SIMPLE>
- **机构：** USC Physical Superintelligence (PSI) Lab
- **论文：** arXiv:2606.08278 — [simple_arxiv_2606_08278.md](../papers/simple_arxiv_2606_08278.md)
- **项目页：** <https://psi-lab.ai/SIMPLE> — [psi-lab-simple.md](../sites/psi-lab-simple.md)
- **入库日期：** 2026-07-16
- **一句话说明：** 基于 AMO/SONIC 栈的人形 loco-manipulation **全栈仿真环境**：MuJoCo+Isaac Sim 双引擎、Gym 评测接口、运动规划与 VR 遥操作数据采集，并集成 Ψ₀、π₀.₅、GR00T 等主流 VLA 评测脚本。

## 仓库要点

- **定位：** full-stack simulation environment for humanoid loco-manipulation；非单篇算法 repo，而是 **benchmark + 数据生成 + 策略集成** 基础设施。
- **策略集成：** 原生支持 **Psi0、Pi05、GR00T** 等主流 VLA；README 含各任务 L0/L1/L2 成功率对照表。
- **评测数据：** OOD 评测环境托管于 Hugging Face [`USC-PSI-Lab/psi-data`](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval)。
- **任务示例：** `G1WholebodyXMovePickTeleop-v0`、`G1Wholebody BendPickMP-v0`、`G1Wholebody HandoverTeleop-v0` 等 G1 全身任务族。
- **底层运控：** 构建于 **AMO/SONIC** 全身 tracking；高层 VLA 输出 kinematic trajectory + base command。

## 开源状态

- **已开源**（截至 2026-07-16）：训练/评测/数据采集代码公开；评测资产与部分数据在 HF `psi-data`。
- 与论文 Abstract「We will open-source our entire codebase」一致；以仓库实际目录为准复现。

## 对 wiki 的映射

- [SIMPLE 论文实体](../../wiki/entities/paper-loco-manip-161-075-simple.md)
- [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)
- [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
