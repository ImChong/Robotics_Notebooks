# EmbodiedIntentBench（arXiv:2611.10498）

> 来源归档（paper）

- **标题：** EmbodiedIntentBench: Benchmarking Intent-Driven Runtime Adaptation for Mobile Robotic Missions
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.10498>
- **PDF：** <https://arxiv.org/pdf/2611.10498v1>（v1，2026-11-07，cs.RO；首页自注「Paper is currently under review」）
- **代码：** https://github.com/rajatbh21/EmbodiedIntentBench-artifact
- **作者 / 机构（论文首页）：** Rajat Bhattacharjya、Changwoon Lee、Hyeonjong Ju、Sing-Yao Wu、Hamidreza Alikhani、Zhuoyi Yang、Ian Harris、Nalini Venkatasubramanian、Nikil Dutt；University of California, Irvine（美国）与 Yonsei University（韩国）
- **入库日期：** 2026-11-10
- **一句话说明：** 面向移动机器人（无人机）任务的基准：把覆盖、时效、证据、汇报与资源写成「任务合同」，在固定仿真轨迹回放里让计算决策真实占用时间与电量并挤掉后续观测，衡量任务整体是否完成。
- **核对范围：** 通读 v1 PDF 正文第 1–7 节（含图 1–6、表 1–4 的文字与数值）、附录 A（A.1–A.6，含表 5）；参考文献列表只扫读未逐条核对；图像本身未逐像素检视，数字以正文 / 图注 / 表中文字为准；artifact 仓库只读了 README 前半部分（概述、上手命令、纳入范围表开头），未读 docs/ARTIFACT.md、REPRODUCING.md、VERIFICATION.md，未运行任何代码。
- **通读：** 已完成

## 开源状态

- 公开 artifact 仓库（rajatbh21/EmbodiedIntentBench-artifact）：README 称包含动态 Lake 研究的录制数据、预测库、执行配置、九种策略、参考结果与验证代码；多阶段 Lake / Depot 的完整录制与预测库**未包含**（论文脚注 1：接收后发布）。README 另称 Linux 为文档环境，Windows 复跑任务成功字段一致但通信与能耗字段在 18 个远程路由任务上未达精确一致，原因未明。

## 核心摘录

1. **合同形式（§2.2，式 1）：** C = (G, D, Q, B)，分别为目标与证据、时间约束、汇报义务、资源约束；JSON 规范，可直接给或由「支持的」运营描述构造，基准运行时不假设开放式自然语言理解。
2. **因果执行（§2.4，式 5）：** 串行阻塞处理；下一个可处理观测为 j = min{k>i | t_k ≥ f_i}，期间采集的帧记为「未处理机会」不排队；迟到的结果仍计入已消耗资源。
3. **任务成功（§2.5，式 7）：** S(H_T, C) 为合同全部要求的合取；诊断量含覆盖、确认、及时/唯一汇报、错误/重复发现、请求超时、处理/跳过观测数、资源消耗。
4. **多阶段研究（§4.1、§5.1、表 2–3）：** Lake 6 段、Depot 4 段录制，每段 120 s、60 帧（960×720，每 2 s 一帧）；Depot 上 Mask R-CNN 双视角并集全库召回 0.875（680 个可见目标观测 / 240 个潜在帧），16 个任务仅成功 1；分块 YOLO11s 召回 0.807，成功 14/16。并集配置标称推理延迟 6.91 s，约每 4 个观测机会才处理 1 个，64 个必需阶段只完成 22 个，电量储备合规 8/16。
5. **动态紧迫度研究（§5.2）：** 9 种策略 × 72 条件（6 录制 × 3 紧迫度 × 2 网络 × 2 电量），对比子集 48 条件；紧迫度感知质量+风险路由 25/48，盲版 24/48，固定远程 Mask R-CNN 22/48；迟到请求 90→0，建模机载任务能耗 198→168 J（约 15%，不含飞行）。
6. **因果消融（§5.3）：** 原始延迟下因果回放 218 次成功，无跳过（no-skip）反事实 255；两策略对比 41/72 vs 38/72 在 no-skip 下变为 45/72 vs 48/72（顺序反转）；推理延迟加倍时因果成功 218→186，no-skip 保持 255。no-skip 是分析性消融，不是可实现的并行实现。
7. **Jetson 与意图驱动停止规则（§5.1、§5.5、附录 A.5–A.6）：** Jetson AGX Orin（MODE_15W，JetPack 6.1，FP32）CPU/GPU 剖面下，每个自适应路由器 26/72（CPU）与 60/72（GPU）；固定本地 Mask R-CNN 0/72 与 60/72；固定远程 38/72。停止规则使固定本地 Mask R-CNN 估计任务能耗在 2 / 3 观测确认下少 28% / 20%，成功数不变（60/72）；5184 个配对重放中无成功变失败、3 个失败变成功（A.6）；上机执行回放选出的计划，整段模块能耗均值降 7.7%（A.6：323.7→298.9 J）。
8. **作者自述范围（§1、§7）：** 评测隔离「计算决策」，固定录制轨迹，导航与物理控制在范围之外；未来工作为闭环执行与实测边云部署。

## 对 wiki 的映射

- [paper-embodiedintentbench](../../wiki/entities/paper-embodiedintentbench.md)
