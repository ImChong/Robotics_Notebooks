# Praxis-1: From Video Pretraining to Robot Control（Runway，2026-11）
> 来源归档（官方技术报告） · 版本：Runway Research，November 2026 · 核查日期：2026-11-12

- **原文：** https://runway.com/research/praxis-1-video-pretraining-to-robot-control
- **发布说明：** https://runway.com/news/company-news/praxis-1-is-now-available（2026-11-11）
- **项目/代码：** https://github.com/runwayml/praxis-1-oss
- **权重集合：** https://huggingface.co/collections/runwayml/praxis-1
- **机构：** Runway
- **类型：** technical report（网页全文；该来源未指定 arXiv 编号，不虚构 PDF）
- **核对范围：** 技术报告 §1–§13（引言、架构、训练阶段、scaling、推理、仿真与真机评测、YAM/Franka、RT-1/SO-101、新本体适配、guidance、离线评估、NVFP4、局限和开放计划），以及官方发布说明。
- **通读：** 已完成

## 研究问题与思路

机器人动作标注昂贵，海量通用视频可先学习物体动力学、交互与未来场景变化。Praxis-1 以 26.9B MoE 视频骨干为基础、314M 参数的动作专家接收视频编码器 8 层深度特征 taps，联合同步预测未来视频 latent 与动作块；共享 backbone/expert，机器人本体通过小型输入输出投影切换。总计约 27.2B 参数（bf16 权重约 54.4 GB）。

## 训练与推理机制

1. 视频基础预训练：12 种视频/图像/音频相关生成任务，全序列注意力。
2. 具身视频中训练：切换 chunk-causal 注意力，主要使用人类第一视角操作视频，加机器人、通用与少量游戏视频；teacher forcing。
3. 视频+动作联合中训练：新增动作专家，机器人示教监督动作与视频，两个 rectified-flow velocity loss 等权；训练中保留约 30% 只含视频的任务。
4. 本体微调：从基础 checkpoint 继承共享骨干与动作专家，新机器人重新初始化其投影，并保持视频 loss。

- 输入：文本任务、相机观测、动作窗口前的机器人绝对位姿，以及执行后的历史视频。
- 输出：视频 latent 与一段动作；通常 24 行 × 15 Hz = 1.6 秒，窗口首次调用为 21 行。
- 默认 2 次联合 Euler 去噪，DROID 配置视频 CFG=4，t_shift=12.5；后者并非跨本体通用超参。
- 单 H100 上默认分块调用往返约 0.55–0.6 秒；RTC 为可选模式，不能把推理服务时延误写成动作控制周期。
- 单臂 Franka 行动表征 10D（xyz + 6D 旋转 + 夹爪），YAM 双臂 20D；不同机器人使用不同投影。

## 报告中的评测与边界

| 测试 | 报告结果 | 边界 |
|---|---|---|
| RoboLab-120 | 555/1200 = 46.2% | DROID 微调，120 任务各 10 次；非所有模型中的绝对第一 |
| Franka FR3 真机 vs FLUX 3 Action | 88/100 vs 80/100 | 同一场景十项任务，每项十次，2026-11-02 |
| Google RT-1 / SimplerEnv | 80.0% | 未在预/中训练见过 RT-1，本体微调后模拟测试 864 episodes |
| Noble Machines | 训练过布局 15/15，陌生仓储 6/10 | 29 分钟示教微调，**仅仿真**，未宣称已上物理机器人 |
| SO-101 | 约 1,070 小时公开数据微调 | 官网展示真机片段，但不等于所有任务统计成功率 |
| NVFP4 | 279/600 vs bf16 287/600 | 报告的量化感知微调实验；不是任意 PTQ 或硬件验证 |
| YAM RTC | 24/30 vs 分块 28/30；中位成功时间 9.1 vs 15.2 秒 | 同一五物体放入袋子的实验，RTC 以速度换部分成功率 |

## 作者明确的局限

- 对相似物体和含糊指令容易选错目标，偏差常已出现在想象的未来视频。
- 单提示自发执行复杂多阶段任务仍不稳；每阶段单独指令可改善。
- Oracle 未来视频条件下动作末端误差约为部署联合预测的 1/5，说明视频预测质量仍是瓶颈；这一开环指标不能直接等同真实任务成功率。
- 论文中的多个对照为作者测量，不应把单一基准的成绩推广为任意场景的泛化保证。

## 对 wiki 的映射

- [Praxis-1 统一实体](../../wiki/entities/runway-praxis-1.md)。
- [官方研究/项目页记录](../sites/runway-praxis-1.md)。
- [官方源码阅读与复现入口](../repos/runway-praxis-1-oss.md)。
