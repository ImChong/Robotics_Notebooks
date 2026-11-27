# ActiveArena: Benchmarking and Understanding Active Perception in Robotic Manipulation

- **作者：** Yibo Li, Enshen Zhou, Rui Chen, Yanjun Ding, Mengzhen Liu, Yi Han, Jiabo Zhan, Lipeng Wang, Shanghang Zhang, Lu Sheng
- **机构：** 北京航空航天大学（Beihang University）、北京智源人工智能研究院（BAAI）、北京大学（Peking University）、清华大学（Tsinghua University）
- **版本 / 载体：** arXiv 预印本 `arXiv:2609.24124v2 [cs.RO]`（2026-09-21 提交 v1，2026-09-23 修订 v2，43 页）
- **原文：** https://arxiv.org/html/2609.24124 （摘要页：https://arxiv.org/abs/2609.24124 ）
- **项目页 / 代码：** 已开源 — 项目主页 [ActiveArena](https://leeibo.github.io/ActiveArena)，仿真与基准仓库 [leeibo/ActiveArena](https://github.com/leeibo/ActiveArena)（MIT 协议），模型训练与推理仓库 [leeibo/ActiveArena-VLA](https://github.com/leeibo/ActiveArena-VLA)，数据集 [leeibo/ActiveArena-Data](https://huggingface.co/datasets/leeibo/ActiveArena-Data) 与资产 [leeibo/ActiveArena-Assets](https://huggingface.co/datasets/leeibo/ActiveArena-Assets) 均已公开；公开 3 组 OFT 检查点权重（[leeibo/ActiveArena-VLA](https://huggingface.co/leeibo/ActiveArena-VLA)）
- **核对范围：** 通读 arXiv HTML 全文（第 1–5 节与附录 A–H，共 2,310 行）、项目主页全文（`project_page.txt`）以及官方双仓库 `README.md` 与目录样本（`files_sample`），核对表 1–20 的核心数值、13 组 ActiveArena-VLA 消融矩阵、18 维动作空间定义、关键帧公式（式 7–21）与真机 Astribot S1 评测数据
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与三类缺口：** 论文指出既有具身操作基准普遍忽视了主动感知（Active Perception）的两大核心维度——主动信息获取（Active Information Acquisition）与主动信息维护（Active Information Maintenance），根源在于仿真器的视角固定（embodiment gap）、桌面局促（scene gap）与交互受限（interaction gap）。
2. **ActiveArena-Sim：** 基于 RoboTwin 2.0 扩展，引入 Astribot S1 双臂人形本体（双 7-DoF 臂 + 2 夹爪 + 1-DoF 躯干偏航 `astribot_torso_joint_4` + 1-DoF 头部俯仰 = 18 维动作向量），将可见场景覆盖提升 6.3 倍；工作空间扩展为半径 0.30–1.08 m 的 $180^\circ$ 环形扇面及双层架构（交互面积 2 倍、信息获取能力 2 倍）；仅暴露动态头部相机（$512 \times 384$，$60^\circ$ VFOV）给策略，并在线记录帧级过程标注（`subtask`、`stage`、可见/已发现掩码与关键帧标记）。
3. **ActiveArena-Bench：** 包含 2 大任务族（视觉探索、交互式信息获取）、5 个细分类别（SS 5 项、SL 16 项、ML 6 项、MD 3 项、IA 5 项，共 35 项任务），每项任务提供 100 条 ID 演示轨迹（总计 581.2k 帧 / 10.76 小时）与严格分离的 50 种子 ID/OOD 评测协议（OOD 引入未见背景、光照、更密集杂物及他任务目标干扰物）。
4. **ActiveArena-VLA 与核心实验结论：** 基于 StarVLA 与 `Qwen3-VL-2B-Instruct` 构建 13 组对照配置。主榜单（表 2）中 `ActiveArena-OFT`（12 帧动作块均匀采样 + 18-D 本体状态 + 子任务辅助监督）在 35 项任务上取得 62.97% ID / 47.60% OOD 成功率，显著领先外部基线 `MemER`（35.31% / 23.43%）、`FAST-WAM`（35.60% / 2.06%）、`MemoryVLA`（22.23% / 11.20%）、`SaPaVe`（20.91% / 10.51%）、$\pi_{0.5}$（16.86% / 1.71%）与 `HiF-VLA`（10.69% / 6.57%）；固定视角消融 `ActiveArena-Fixed` 仅 0.34% ID / 0.17% OOD，Discovery Miss 达 96.3%/97.3%。
5. **规划器记忆管理与真机迁移：** `ActiveArena-Plan` 使用微调的 `Qwen3-VL-2B` 规划器在 12 帧步长-16 滚动历史中预测当前子任务并检索稀疏子任务切换帧，以 60.97% ID / 45.54% OOD 逼近全量记忆模型，同时将 VLA 训练峰值显存降低 46%（78.9 GB → 42.6 GB）。8 项 Astribot S1 真机实验（640 条轨迹、每项 20 次扰动测试）验证了趋势一致性：`ActiveArena-OFT` 平均成功率 55.63%，优于 `MemER`（34.38%）、`SaPaVe`（24.38%）和 $\pi_{0.5}$（23.75%）。
6. **核对勘误记录：** 对比原文表 2 / 表 5 与项目页 HTML 表格发现，项目页表格中 `MD` 与 `IA` 两列数值存在列标题与数据互换；按表 5 公布的各类别任务数（5 SS, 16 SL, 6 ML, 3 MD, 5 IA）验算加权平均，原文表 2 数值（如 `ActiveArena-OFT` 的 MD 44.00/35.33、IA 31.60/27.20）与总平均 62.97/47.60 严格自洽，知识条目采用原文表 2 核定数据。

- 对 wiki 的映射：[ActiveArena: Benchmarking and Understanding Active Perception in Robotic Manipulation](../../wiki/entities/paper-agentic-2609-24124.md)
