---
type: overview
tags: [robot-learning, vla, manipulation, navigation, embodied-eval]
status: complete
updated: 2026-11-16
original:
  - ../../sources/blogs/wechat-robotics-29-papers-2026-11-16.md
sources:
  - ../../sources/blogs/wechat-robotics-29-papers-2026-11-16.md
related:
  - ../entities/paper-actioncodec2-streaming-action-tokenizer.md
  - ../entities/paper-maniladder-manipulation-benchmark.md
summary: "2026-11-16 具身智能导览 29 篇论文的独立详情索引，按机器人策略、示教、仿真、评测与系统工程区分方法、实验及开放边界。"
---

# 2026-11-16 具身智能前沿论文 29 篇｜29 Papers in Embodied AI

## 一句话定义

本页按原附件阅读顺序整理 **29 篇不同的原始论文**；每篇跳转独立详情节点，避免把二手导览当作已读的一手论文或单篇研究的综合结论。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉语言动作模型及执行接口 |
| IL | Imitation Learning | 利用示教数据学习机器人策略 |
| Sim2Real | Simulation to Reality | 仿真到真实机器人部署迁移 |
| OOD | Out-of-Distribution | 未见任务及测试条件 |

## 为什么重要

同一批论文分别报告 **真实任务进度、成功率、动作重建误差、符号规划评分、图像质量** 等不同指标，不能跨任务直接排名。阅读时优先区分：动作表示优化与真实闭环控制、仿真结果与真机结果、公开完整系统与仅公开子模块。

## 核心原理

### 阅读优先级

- **先读策略输出与接口**：[ActionCodec2](../entities/paper-actioncodec2-streaming-action-tokenizer.md) 分析边解码边行动的表示瓶颈；[JustRUA](../entities/paper-justrua-robot-use-agents.md) 对照动作步长、历史和感知工具的 Agent 接口。
- **再读可比性与困难任务**：[ManiLadder](../entities/paper-maniladder-manipulation-benchmark.md) 用统一训练预算的四级难度校准任务，帮助发现平均分掩盖的能力缺口。
- **按工程目标扩展**：运动控制看 BlinkMimic / Elastra，示教与人机接力看 HALO / VIM / CrowdMAPS，仿真与数据生成看 Alloy / BWM-Forge，真实系统集成看 ARTS / SPARC / Hibikino-Musashi@Home。

### 本期 29 篇独立论文详情

| # | 论文 | arXiv | 方向 |
|---:|---|---|---|
| 1 | [ActionCodec2](../entities/paper-actioncodec2-streaming-action-tokenizer.md) | [2611.15490](https://arxiv.org/abs/2611.15490) | 动作编解码 |
| 2 | [JustRUA](../entities/paper-justrua-robot-use-agents.md) | [2611.16009](https://arxiv.org/abs/2611.16009) | 机器人智能体接口 |
| 3 | [ManiLadder](../entities/paper-maniladder-manipulation-benchmark.md) | [2611.16374](https://arxiv.org/abs/2611.16374) | 操作评测 |
| 4 | [BlinkMimic](../entities/blinkmimic.md) | [2611.16422](https://arxiv.org/abs/2611.16422) | 人形动作跟踪 |
| 5 | [HALO](../entities/paper-halo-2611-16439.md) | [2611.16439](https://arxiv.org/abs/2611.16439) | 示教采集 |
| 6 | [Autonomous Trail Racing](../entities/paper-bicycle-trail-racing-2611-15280.md) | [2611.15280](https://arxiv.org/abs/2611.15280) | 移动导航 |
| 7 | [VIM](../entities/paper-vim-operator-intent-2611-16419.md) | [2611.16419](https://arxiv.org/abs/2611.16419) | 遥操作意图 |
| 8 | [Alloy](../entities/paper-alloy-multimaterial-2611-16312.md) | [2611.16312](https://arxiv.org/abs/2611.16312) | 物理仿真 |
| 9 | [NSDL](../entities/paper-nsdl-multirobot-2611-16671.md) | [2611.16671](https://arxiv.org/abs/2611.16671) | 多机器人规划 |
| 10 | [BAMOS](../entities/paper-bamos-search-2611-15689.md) | [2611.15689](https://arxiv.org/abs/2611.15689) | 主动搜索 |
| 11 | [STVLA](../entities/paper-stvla-spatial-temporal-memory.md) | [2611.16589](https://arxiv.org/abs/2611.16589) | VLA 空间记忆 |
| 12 | [FLUX-mimic](../entities/paper-flux-mimic-2611-15974.md) | [2611.15974](https://arxiv.org/abs/2611.15974) | 工业视频动作 |
| 13 | [CrowdMAPS](../entities/paper-crowdmaps-2611-15078.md) | [2611.15078](https://arxiv.org/abs/2611.15078) | 模仿学习数据 |
| 14 | [ARTS](../entities/paper-arts-surgery-2611-15135.md) | [2611.15135](https://arxiv.org/abs/2611.15135) | 手术研究平台 |
| 15 | [Elastra](../entities/paper-elastra-2611-15112.md) | [2611.15112](https://arxiv.org/abs/2611.15112) | 弹性材料仿真 |
| 16 | [Robo-SWE](../entities/paper-robo-swe-2611-16116.md) | [2611.16116](https://arxiv.org/abs/2611.16116) | 机器人软件智能体 |
| 17 | [DexRemo](../entities/paper-dexremo-2611-15981.md) | [2611.15981](https://arxiv.org/abs/2611.15981) | 力觉灵巧操作 |
| 18 | [Passive Soft Extensions](../entities/paper-passive-soft-extensions-2611-15386.md) | [2611.15386](https://arxiv.org/abs/2611.15386) | 软体控制 |
| 19 | [VINE](../entities/paper-vine-exocentric-nav-2611-16282.md) | [2611.16282](https://arxiv.org/abs/2611.16282) | 外部摄像头导航 |
| 20 | [Reins](../entities/paper-reins-robot-harness.md) | [2611.16632](https://arxiv.org/abs/2611.16632) | 长程机器人智能体 |
| 21 | [BWM-Forge](../entities/paper-bwm-forge-2611-16221.md) | [2611.16221](https://arxiv.org/abs/2611.16221) | 世界编程 |
| 22 | [VOKE](../entities/paper-voke-2611-16423.md) | [2611.16423](https://arxiv.org/abs/2611.16423) | 掌内运动学 |
| 23 | [GenGenIK](../entities/paper-gengenik-2611-15565.md) | [2611.15565](https://arxiv.org/abs/2611.15565) | 逆运动学 |
| 24 | [PreNav](../entities/paper-prenav-2611-15093.md) | [2611.15093](https://arxiv.org/abs/2611.15093) | 开放词汇导航 |
| 25 | [SPARC](../entities/paper-sparc-2611-15443.md) | [2611.15443](https://arxiv.org/abs/2611.15443) | 救援机器人系统 |
| 26 | [MOSAIC](../entities/paper-mosaic-clutter-2611-15937.md) | [2611.15937](https://arxiv.org/abs/2611.15937) | 杂乱操作避障 |
| 27 | [SO-101 Toilet Cleaning](../entities/paper-so101-toilet-cleaning-2611-15381.md) | [2611.15381](https://arxiv.org/abs/2611.15381) | 长程移动操作 |
| 28 | [Physical Action Model Training](../entities/paper-physical-action-training-recipe-2611-16450.md) | [2611.16450](https://arxiv.org/abs/2611.16450) | 行动模型训练理论 |
| 29 | [Hibikino-Musashi@Home](../entities/paper-hibikino-home-2026-2611-16297.md) | [2611.16297](https://arxiv.org/abs/2611.16297) | 服务机器人系统 |

## 工程实践

1. **核对身份与版本**：同一 arXiv ID 只对应一份实体详情；已存在的 JustRUA / STVLA / Reins 沿用原页面。
2. **明确结果分母**：动作重建轨迹误差不等于物理闭环误差；任务进度比例不等于成功率；仿真结果不能直接写成真机稳定性。
3. **评估开源范围**：有可运行官方代码时在实体页列出实际入口和模块；仅开放数据、模型、部分子模块或尚未公开完整代码时写明限制，不伪造运行时序图。
4. **建立研究复现入口**：从论文原文及项目页进入，再阅读对应独立节点的流程图、实际源码时序、评测与局限。

## 局限与风险

- 本索引的初始素材是第三方 PDF 导览，**不是** 29 项实验的独立复现。每篇的主张均需以其官方论文、项目与代码为准。
- 有些项目只公开局部源码、数据或模型；源节点健康度 **3/3 只反映结构化来源、阅读骨架与原文标记**，不等于论文推理/真机实验已复现。
- 论文版本可能更新，项目权限和资源公开状态也可能变化。阅读时应以官方原始材料核对当前版本。

## 关联页面

- [ActionCodec2：低延迟动作编码](../entities/paper-actioncodec2-streaming-action-tokenizer.md)
- [ManiLadder：分难度评测](../entities/paper-maniladder-manipulation-benchmark.md)
- [JustRUA：Agent 控制接口](../entities/paper-justrua-robot-use-agents.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)
- [具身评测纵深路线](../../roadmap/depth-embodied-eval.md)

## 参考来源

- [附件对应的原始导览链接及 29 篇身份线索](../../sources/blogs/wechat-robotics-29-papers-2026-11-16.md)
- [ActionCodec2 论文原文](https://arxiv.org/abs/2611.15490)
- [ManiLadder 论文原文](https://arxiv.org/abs/2611.16374)

## 推荐继续阅读

- [ActionCodec2 官方代码](https://github.com/ZibinDong/actioncodec2)
- [ManiLadder 官方项目页](https://maniladder-benchmark.github.io/)
