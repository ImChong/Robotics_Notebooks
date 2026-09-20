# 具身智能入门与实践（二）：代码即策略——当 LLM 拿起键盘驱动机器人

> 来源归档（blog / 微信公众号 · Lumina 机器人技术指南）

- **标题：** 具身智能入门与实践（二）：代码即策略——当 LLM 拿起键盘驱动机器人
- **类型：** blog
- **作者：** 机器人技术指南 / Lumina 具身智能社区
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485113&idx=1&sn=ac3e3e2031a9522e6cb71910d6d1bc5e
- **专辑：** [wechat_lumina_embodied_practice_album](../raw/wechat_lumina_embodied_practice_album_4608355279393816579.md)
- **入库日期：** 2026-09-20
- **抓取方式：** 正文 CAPTCHA；对照 [Embodied-AI-Guide §4 Code 能力 / 3D+LLM](https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/topics/algorithm.md#4-llm-for-robotics--大语言模型在机器人中的应用)
- **一句话说明：** LLM 生成**可执行中间层**（Python 程序或 3D value map），把感知/规划/控制原语组合成可调试策略，比纯文本计划更接近工程落地。

## 核心摘录（归纳）

| 路线 | 代表 | 中间表示 | 本库节点 |
|------|------|----------|----------|
| Code as Policy | CaP | Python 调用感知/运动原语 | [paper-pai-2209-07753-codeaspolicies](../../wiki/entities/paper-pai-2209-07753-codeaspolicies.md) |
| Instruction2Act | I2A | 代码 + 视觉 grounding | [paper-instruction2act](../../wiki/entities/paper-instruction2act.md) |
| VoxPoser | — | 3D value map + motion planner | [paper-voxposer](../../wiki/entities/paper-voxposer.md) |
| OmniManip | — | 3D 约束 + 操作规划 | [paper-omnimanip](../../wiki/entities/paper-omnimanip.md) |

**工程读法：** 与 [ASPIRE](../../wiki/methods/aspire.md) 的 code-as-policy 持续学习、Anthropic [Embody](../../wiki/entities/anthropic-embody.md) 的「写控制器」接口同族；差异在是否用 frontier LLM 在线写代码 vs 离线技能库。

## 对 wiki 的映射

- [embodied-ai-guide-wechat-album-curator.md](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md) § 篇 2
- [LLM 机器人控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md)
