# 具身智能入门与实践（一）：让机械臂听懂人话——LLM 做任务规划器

> 来源归档（blog / 微信公众号 · Lumina 机器人技术指南）

- **标题：** 具身智能入门与实践（一）：让机械臂听懂人话——LLM 做任务规划器
- **类型：** blog
- **作者：** 机器人技术指南 / Lumina 具身智能社区
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485105&idx=1&sn=6631b3b366acc2cde8e0d84e62070afc
- **专辑：** [wechat_lumina_embodied_practice_album](../raw/wechat_lumina_embodied_practice_album_4608355279393816579.md)
- **入库日期：** 2026-09-20
- **抓取方式：** 正文 URL CAPTCHA；归纳对照 [Embodied-AI-Guide §4 LLM for Robotics](https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/topics/algorithm.md#4-llm-for-robotics--大语言模型在机器人中的应用)
- **一句话说明：** 入门叙事：LLM 在机器人里最可靠的定位是**高层语义理解 + 任务分解 + 工具调用**，与底层控制器或 VLA 执行器分层配合；不是直接输出力矩。

## 核心摘录（归纳）

### 分层栈

| 层 | 职责 | 典型接口 |
|----|------|----------|
| 语言/规划 | 理解指令、分解子任务、选技能 | LLM / VLM planner |
| 几何/约束 | 可行性、碰撞、TAMP | 传统规划器 |
| 执行 | 连续轨迹或 chunk | BC / RL / VLA / 技能库 |

### 文内代表工作 → 本库独立节点

| 工作 | arXiv | 本库节点 | 开源（2026-09-20 核查） |
|------|-------|----------|-------------------------|
| SayCan | 2204.01691 | [paper-saycan](../../wiki/entities/paper-saycan.md) | 部分（google-research 子目录） |
| PaLM-E | 2303.03378 | [paper-palm-e-embodied-language-model](../../wiki/entities/paper-palm-e-embodied-language-model.md) | 权重未公开 |
| EmbodiedGPT | 2305.15021 | [paper-embodiedgpt](../../wiki/entities/paper-embodiedgpt.md) | 待核实 |
| LBYL | 2311.17842 | [paper-look-before-you-leap](../../wiki/entities/paper-look-before-you-leap.md) | 待核实 |
| RT-2 | 2307.15818 | [paper-rt-2](../../wiki/entities/paper-rt-2.md) | 未开源完整训练栈 |
| LLM+P | 2304.11477 | [paper-llm-p](../../wiki/entities/paper-llm-p.md) | 未列官方仓 |
| AutoTAMP | 2306.06531 | [paper-autotamp](../../wiki/entities/paper-autotamp.md) | 待核实 |
| Text2Motion | 2303.12153 | [paper-text2motion](../../wiki/entities/paper-text2motion.md) | 待核实 |
| OpenVLA | 2406.09246 | [paper-openvla](../../wiki/entities/paper-openvla.md) | **已开源** |
| Octo | 2405.12213 | [paper-octo](../../wiki/entities/paper-octo.md) | **已开源** |

## 对 wiki 的映射

- 阅读坐标：[embodied-ai-guide-wechat-album-curator.md](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md) § 篇 1
- 概念：[LLM 机器人控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md)
- 方法：[SayCan](../../wiki/methods/saycan.md)、[VLA](../../wiki/methods/vla.md)
