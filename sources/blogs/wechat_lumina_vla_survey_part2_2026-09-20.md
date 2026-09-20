# VLA综述：具身智能路线梳理（二） - 建议收藏

> 来源归档（blog / 微信公众号 · Lumina 机器人技术指南）

- **标题：** VLA综述：具身智能路线梳理（二） - 建议收藏
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzA5NTM1OTgxNA==&mid=2247485308&idx=1&sn=bc5dbaad64792c7aabc5352276ca1abd
- **专辑：** [wechat_lumina_embodied_practice_album](../raw/wechat_lumina_embodied_practice_album_4608355279393816579.md)
- **入库日期：** 2026-09-20
- **对照来源：** [Embodied-AI-Guide §5.2–5.3 分层双系统 & 2025 工作](https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/topics/algorithm.md#52-分层双系统-vla202505-更新)
- **一句话说明：** 2025 主线是 **System 2 慢推理 + System 1 快控制** 的分层 VLA，以及 world model / 3D / 安全对齐与产业级部署系统并行演进。

## 核心摘录

### 分层双系统（System 2 / System 1）

| 维度 | 典型差异 | 代表 |
|------|----------|------|
| 架构 | 单模型 vs 双模型 | π 系列 vs Hi-Robot |
| 中间表示 | 语言 / 子目标 / latent | 影响可控性与频率分工 |
| 部署 | 真机频率、全身控制 | GR00T、Gemini Robotics、Helix |

### 产业/系统级节点（复用已有实体）

| 系统 | 本库节点 |
|------|----------|
| Figure Helix | [helix-25](../../wiki/entities/helix-25.md) |
| NVIDIA GR00T N1 | [isaac-gr00t](../../wiki/entities/isaac-gr00t.md) |
| Gemini Robotics | [gemini-robotics](../../wiki/entities/gemini-robotics.md) |
| Physical Intelligence openpi | [paper-pi0](../../wiki/entities/paper-pi0.md) |
| π0.5 | 见 [pi07-policy](../../wiki/methods/pi07-policy.md) / π 系列页 |

### 2025 代表论文（独立节点）

| 工作 | 本库节点 |
|------|----------|
| WorldVLA | 待升格（文内链接；图谱见 [vla.md](../../wiki/methods/vla.md)） |
| UniVLA | 待升格 |
| HybridVLA | 待升格 |
| DexGraspVLA | [paper-dexholdem](../../wiki/entities/paper-dexholdem.md) 邻域 / 待升格 |
| BridgeVLA | 待升格 |

**原则：** 已有 `paper-*` / 实体页只回链；无页标「待升格」，不在此 ingest 重复造页。

## 对 wiki 的映射

- [embodied-ai-guide-wechat-album-curator.md](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md) § VLA 二
- [VLA 演进纵览](../../wiki/overview/vla-evolution-lineage.md)
