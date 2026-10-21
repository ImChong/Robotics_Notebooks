# Thinker-WM 论文来源归档

> 来源归档（paper；核查日期：2026-10-16）

- **标题：** Thinker-WM: A Unified World Model with Shared Multimodal Representations for Embodied Intelligence
- **简称：** Thinker-WM
- **类型：** arXiv 预印本
- **作者：** Thinker Team, Peng Gao, Jichao Jiao, Chen Liao, Daqin Luo, Yiming Ma, Baiyu Pan, Hailin Shi, Jiyuan Wang, Xiangming Xu, Xiongwei Zhao
- **机构：** UBTECH Robotics（论文团队署名）
- **版本 / 日期：** v1；2026-10-13；cs.AI
- **论文：** <https://arxiv.org/abs/2610.18447>
- **PDF：** <https://arxiv.org/pdf/2610.18447>
- **项目页：** <https://thinkercosmos.ubtrobot.com/thinker-wm/>（[归档](../sites/thinker-wm.md)）
- **代码：** <https://github.com/UBTECH-Robot/Thinker-WM>（[归档](../repos/thinker-wm.md)）
- **模型：** <https://huggingface.co/UBTECH-Robotics/Thinker-WM>（[归档](../models/thinker-wm.md)）
- **入库日期：** 2026-10-16

## 论文要点

Thinker-WM 将 RGB 视频、度量深度与机器人动作放进共享的中间动态表征，由统一动态 Transformer（Unified Dynamics Transformer）共同演化表征，再由各模态专属的编码器和解码器处理输入输出。论文提出：外观、几何和动作可在共享表征中联合建模，同时保留各自不同的输出结构。

同一套参数支持六类任务：视频与动作联合生成、世界预测、策略推理、动作条件生成、视频到动作，以及逆动力学。作者还用冻结视觉表征上的轻量 motion probe 检查运动信息是否可恢复。

## 数据与评测（论文摘要口径）

- 训练数据约 **20K 小时第一视角视频、10K 小时真机数据、1K 小时仿真数据**；其中约 **4K 小时真机数据带有对齐的深度监督**。
- 论文称 Thinker-WM 在 **EWMBench、WorldOlympiad、WorldArena** 三项具身世界模型评测中排名第一。
- 具体排名与分数依赖各基准的评测设置；对比不同榜单结果时应核对任务、指标及版本。

## 阅读边界

- 本页仅记录论文摘要与官方发布入口，不复述全文实验表格；逐项指标和消融结论以论文原文为准。
- 论文报告的训练语料规模，与 GitHub README 的简短介绍口径并不完全一致；此处按论文摘要记录。
- 论文与项目发布于 2026 年 10 月；代码仓库 README 仍有旧占位文本和引用元数据，详见仓库归档。

## Robotics_Notebooks 映射

- Wiki 实体：[`wiki/entities/paper-thinker-wm-unified-world-model.md`](../../wiki/entities/paper-thinker-wm-unified-world-model.md)

## 一手来源

- 论文：[arXiv:2610.18447](https://arxiv.org/abs/2610.18447)
- 项目页：<https://thinkercosmos.ubtrobot.com/thinker-wm/>
- 代码：<https://github.com/UBTECH-Robot/Thinker-WM>
