# 近三年世界模型 3 大技术路线！这 15 个项目全开源（推荐收藏）

> 来源归档（blog / 微信公众号）

- **标题：** 近三年世界模型 3 大技术路线！这 15 个项目全开源（推荐收藏）
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg
- **发表日期：** 2026-06-03
- **入库日期：** 2026-06-03
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）；正文约 8800 字 / 26 图
- **原始落盘：** [wechat_world_models_15_2026-06-03.md](../raw/wechat_world_models_15_2026-06-03.md)
- **参考资料索引：** [shenlan_world_models_15_reference_catalog.md](../papers/shenlan_world_models_15_reference_catalog.md)
- **姊妹篇：** [机器人世界模型训练闭环 taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)（arXiv:2605.00080 综述）；[Ego 9 篇 · 世界模型分类](../../wiki/overview/ego-category-03-world-models.md)
- **一句话说明：** 深蓝具身智能策展 15 个高引开源世界模型项目，按 **级联（6）/ 联合（6）/ 虚拟沙盒（3）** 三条路线组织，强调「什么样的预测才值得托付一次真实物理交互」与可复现代码基线。

## 核心摘录（归纳，非全文）

### 三条路线 × 15 项目

| 路线 | 篇数 | 分类 hub | 核心机制（归纳） |
|------|------|----------|------------------|
| **01 级联架构** | 6 | [world-models-route-01-cascade](../../wiki/overview/world-models-route-01-cascade.md) | 先预测未来视觉/潜特征，再解码动作；误差在级联间传递 |
| **02 联合架构** | 6 | [world-models-route-02-joint](../../wiki/overview/world-models-route-02-joint.md) | 未来与动作在同一扩散/自回归骨干中联合建模 |
| **03 虚拟沙盒** | 3 | [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) | WM 作 RL/评估环境，想象 rollout 替代昂贵真机试错 |

### 01 级联（6）

| # | 工作 | venue | 引用（文内） | Wiki |
|---|------|-------|-------------|------|
| 01 | TesserAct | ICCV 2025 | 75 | [paper-shenlan-wm-01-tesseract](../../wiki/entities/paper-shenlan-wm-01-tesseract.md) |
| 02 | VPP | ICML 2025 | 200 | [paper-shenlan-wm-02-vpp](../../wiki/entities/paper-shenlan-wm-02-vpp.md) |
| 03 | LaPA | ICLR 2025 | 252 | [paper-shenlan-wm-03-lapa](../../wiki/entities/paper-shenlan-wm-03-lapa.md) |
| 04 | mimic-video | — | 29 | [mimic-video 方法页](../../wiki/methods/mimic-video.md) |
| 05 | villa-X | — | 58 | [paper-shenlan-wm-05-villa-x](../../wiki/entities/paper-shenlan-wm-05-villa-x.md) |
| 06 | Video Generators are Robot Policies | — | 41 | [paper-shenlan-wm-06-video-gen-robot-policies](../../wiki/entities/paper-shenlan-wm-06-video-gen-robot-policies.md) |

### 02 联合（6）

| # | 工作 | venue | 引用 | Wiki |
|---|------|-------|------|------|
| 07 | WorldVLA / RynnVLA-002 | — | 180 | [paper-shenlan-wm-07-worldvla](../../wiki/entities/paper-shenlan-wm-07-worldvla.md) |
| 08 | UWM | RSS 2025 | 102 | [paper-shenlan-wm-08-uwm](../../wiki/entities/paper-shenlan-wm-08-uwm.md) |
| 09 | GR-1 | ICLR 2024 | 358 | [paper-shenlan-wm-09-gr1](../../wiki/entities/paper-shenlan-wm-09-gr1.md) |
| 10 | UVA | RSS 2025 | 140 | [paper-shenlan-wm-10-uva](../../wiki/entities/paper-shenlan-wm-10-uva.md) |
| 11 | Cosmos Policy | — | 61 | [paper-shenlan-wm-11-cosmos-policy](../../wiki/entities/paper-shenlan-wm-11-cosmos-policy.md) |
| 12 | F1-VLA | — | 24 | [paper-shenlan-wm-12-f1-vla](../../wiki/entities/paper-shenlan-wm-12-f1-vla.md) |

### 03 虚拟沙盒（3）

| # | 工作 | venue | 引用 | Wiki |
|---|------|-------|------|------|
| 13 | DreamerV3 | Nature | 1475 | [paper-shenlan-wm-13-dreamerv3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md) |
| 14 | RLVR-World | NeurIPS 2025 | 34 | [paper-shenlan-wm-14-rlvr-world](../../wiki/entities/paper-shenlan-wm-14-rlvr-world.md) |
| 15 | WorldGym | — | 13 | [paper-shenlan-wm-15-worldgym](../../wiki/entities/paper-shenlan-wm-15-worldgym.md) |

### 文内收束判断（策展）

- **评价划线：** 除视觉保真外，应考察 **控制一致性、物理一致性、下游任务增益**——与 [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) 一致。
- **路线渗透：** 级联引入联合训练降误差；联合模型吸收沙盒闭环自演进——非互斥替代。
- **引用数据：** 截止 2026-06-02，仅作参考，非研究水平排名。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [world-models-15-open-source-technology-map](../../wiki/overview/world-models-15-open-source-technology-map.md) | **父节点** |
| [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) | 学术综述三线 taxonomy 对照 |
| [generative-world-models](../../wiki/methods/generative-world-models.md) | 生成式 WM 方法链 |
| [world-action-models](../../wiki/concepts/world-action-models.md) | WAM 概念坐标 |

## 可信度与使用边界

- 15 篇为策展子集，非 exhaustive survey；引用量受发表时间影响。
- 推广/课程信息已剥离；各仓库 License 与权重可得性需自行确认。

## 当前提炼状态

- [x] Agent Reach 抓取与 raw 落盘
- [x] 三线分类 hub + 15 项目索引
- [x] 与既有 WM taxonomy 交叉链接
