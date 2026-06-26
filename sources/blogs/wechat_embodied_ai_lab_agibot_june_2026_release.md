# 智元这轮发布怎么看？五个开源项目和两个能力底座，正在把机器人落地链路拆开

> 来源归档（blog / 微信公众号）

- **标题：** 智元这轮发布怎么看？五个开源项目和两个能力底座，正在把机器人落地链路拆开
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/QWj7F2vhhRrRpX41SaNyaA
- **发表日期：** 2026-06-25
- **入库日期：** 2026-06-26
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install` + 手动安装 [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox））；正文约 0.53 万字 / 7 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **姊妹篇：** [BFM 41 篇运控基座长文](wechat_embodied_ai_lab_bfm_41_papers_survey.md)、[训练栈分层解读](wechat_embodied_ai_lab_robot_training_stack_layers_2026.md)、[世界模型训练闭环](wechat_embodied_ai_lab_robot_world_model_training_loop.md)
- **一句话说明：** 按 **落地链路七段**（数据 → 仿真评测 → 世界模型 → 语义执行 → 运控/感控底座 → 应用交付）解读智元 2026-06 发布：**五个开源项目**（AGIBOT WORLD 2026、Genie Sim 3.0、GE-Sim 2.0、GO-2、Genie Studio Agent）与 **两个能力底座**（BFM-2 运动小脑、AGILE 感控一体）；核心判断：智元正把机器人落地从单点 demo 拆成可复用的分层模块，但各层闭合度仍须真机与长期稳定性验证。

## 核心摘录（归纳，非全文）

### 问题重框

- **不是单模型发布会：** 七段链路覆盖 **数据入口 → 仿真训练评测 → 世界模型 → 语义到动作 → 身体底座 → 工程交付**；与「只讲一个 VLA」的叙事不同。
- **五个开源 + 两个底座：** 开源侧强调 **数据、仿真、世界模型、执行基座（GO-2）、应用编排**；底座侧 **BFM-2**（动作过渡与动态闭环）与 **AGILE**（视觉进入控制闭环）分别补 **小脑** 与 **眼–身接口**。
- **策展态度：** 文内对 GE-Sim 2.0、BFM-2、AGILE 等 **偏谨慎**——方向强，但替代真机交互与长期 sim2real 仍待验证；GO-2 的 Genie Sim Sim2Real **82.9% vs π0.5 77.5%** 等数字引用自材料，须更多任务复现。

### 落地链路七段（文内项目地图）

| 段 | 项目 | 核心问题 |
|----|------|----------|
| 数据入口 | AGIBOT WORLD 2026 | 模仿学习数据是否足够贴近真实部署的物理过程？ |
| 仿真训练与评测 | Genie Sim 3.0 | 场景生成能否进入训练、评测与在线微调闭环？ |
| 世界模型 | GE-Sim 2.0 | 动作条件世界能否响应动作并进入 Eval/RL/Teleop in WM？ |
| 语义到动作执行 | GO-2 | 语义规划如何变成可执行动作序列？ |
| 运动小脑（底座） | BFM-2 | 动作之间如何稳定过渡并形成动态闭环？ |
| 感控闭环（底座） | AGILE | 视觉如何进入控制过程并随反馈调整？ |
| 应用编排与交付 | Genie Studio Agent | 能力如何封装为可编排、可部署、可恢复的企业流程？ |

## 七项目索引

| # | 项目 | 类型 | 链接 |
|---|------|------|------|
| 01 | AGIBOT WORLD 2026 | 开源数据集 | <https://agibot-world.com> · [HF](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) |
| 02 | Genie Sim 3.0 | 开源仿真 | <https://agibot-world.com/genie-sim> · [GitHub](https://github.com/AgibotTech/genie_sim) |
| 03 | GE-Sim 2.0 | 世界模型 | <https://ge-sim-v2.github.io/> |
| 04 | GO-2 | 执行基座 / VLA | [arXiv:2601.11404](https://arxiv.org/abs/2601.11404) · <https://libra-vla.github.io/> |
| 05 | BFM-2 | 能力底座（运控） | [B站视频](https://www.bilibili.com/video/BV1ZzGe6oEmk/) |
| 06 | AGILE | 能力底座（感控） | [B站视频](https://www.bilibili.com/video/BV1SmGD6xEPS/) |
| 07 | Genie Studio Agent | 应用平台 | [微信发布文](https://mp.weixin.qq.com/s/Ha9_0TLyVtec-cL4WqFAbA) |

## 对 wiki 的映射

- [agibot-june-2026-release-technology-map](../../wiki/overview/agibot-june-2026-release-technology-map.md)（**父节点** + Mermaid）
- 子分类 hub：`wiki/overview/agibot-release-category-01-data-entry.md` … `agibot-release-category-06-application-delivery.md`
- 项目实体：`wiki/entities/agibot-world-2026.md`、`genie-sim-3.md`、`ge-sim-2.md`（已有）、`go-2.md`、`agibot-bfm-2.md`、`agibot-agile.md`、`genie-studio-agent.md`
- **BFM-2（智元运控基座）** ≠ **paper-bfm-***（awesome-bfm-papers 学术索引）≠ **BFM 概念页**（行为基础模型综述）

## 可信度与使用边界

- 本文为 **微信公众号策展导读**；技术细节以官方站点、论文、仓库 README 为准。
- BFM-2、AGILE 文内主要依据 **视频标题与简介**（无公开字幕），实体页保持保守归纳。
- 原始抓取正文见 [wechat_agibot_june_2026_release_2026-06-26.md](../raw/wechat_agibot_june_2026_release_2026-06-26.md)。
