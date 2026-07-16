# Lumo-2 官方项目页（Astribot）

> 来源归档（ingest）

- **标题：** Lumo-2: Latent World-Action Model for Predictive, Aligned, and Scalable Robot Learning
- **类型：** site
- **官方入口：** <https://www.astribot.com/research/Lumo2>（英文：<https://www.astribot.com/en/research/Lumo2>）
- **技术报告：** <https://arxiv.org/abs/2607.11270>
- **机构：** Astribot Team（星尘智能）
- **入库日期：** 2026-07-16
- **一句话说明：** Lumo-2 下一代 **潜空间世界–动作模型（latent WAM）**：以物理接地隐空间动力学替代 Lumo-1 显式文本推理；**三阶段渐进模态预对齐**；原生共训 VLM / 野外视频 / 多本体机器人数据；项目页按能力维度聚合 **32 段真机演示视频**。

## 相对 Lumo-1 的三维升级（项目页叙事）

1. **预测推理：** 显式结构化文本规划 → **隐空间世界动力学上的预测推理**（轻量、非像素级生成）。
2. **对齐范式：** 单阶段联合训练 → **三阶段渐进跨模态预对齐**（动力学–动作 → 视觉–语言–动作 → VLWA 共训）。
3. **可扩展性：** 依赖专用机器人示范 → **多源数据原生共训**（VLM 语料、egocentric 人视频、跨本体机器人轨迹），报告更优 scaling 与 OOD 泛化。

## 演示视频索引（OSS 全量）

**CDN 根路径：** `https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/`

路径模式：`{category}/{NN}.mp4`（`NN` 为两位序号）；同名 `.jpg` 为封面。

### Hero

| 文件 | URL |
|------|-----|
| hero.mp4 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/hero.mp4> |

### Collaboration（2）

| # | URL |
|---|-----|
| 01 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/collaboration/01.mp4> |
| 02 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/collaboration/02.mp4> |

### Physical Understanding（4）

| # | URL |
|---|-----|
| 01 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/physical-understanding/01.mp4> |
| 02 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/physical-understanding/02.mp4> |
| 03 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/physical-understanding/03.mp4> |
| 04 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/physical-understanding/04.mp4> |

### Temporal Reasoning（9）

| # | URL |
|---|-----|
| 01 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/temporal-reasoning/01.mp4> |
| 02 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/temporal-reasoning/02.mp4> |
| 03 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/temporal-reasoning/03.mp4> |
| 04 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/temporal-reasoning/04.mp4> |
| 05 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/temporal-reasoning/05.mp4> |
| 06 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/temporal-reasoning/06.mp4> |
| 07 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/temporal-reasoning/07.mp4> |
| 08 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/temporal-reasoning/08.mp4> |
| 09 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/temporal-reasoning/09.mp4> |

### Long-Horizon Task（3）

| # | URL |
|---|-----|
| 01 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/long-horizon-task/01.mp4> |
| 02 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/long-horizon-task/02.mp4> |
| 03 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/long-horizon-task/03.mp4> |

### Dexterous Manipulation（14）

| # | URL |
|---|-----|
| 01 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/01.mp4> |
| 02 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/02.mp4> |
| 03 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/03.mp4> |
| 04 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/04.mp4> |
| 05 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/05.mp4> |
| 06 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/06.mp4> |
| 07 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/07.mp4> |
| 08 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/08.mp4> |
| 09 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/09.mp4> |
| 10 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/10.mp4> |
| 11 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/11.mp4> |
| 12 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/12.mp4> |
| 13 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/13.mp4> |
| 14 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/lumo2/dexterous-manipulation/14.mp4> |

## 核心摘录（面向 wiki 编译）

### 1) 三阶段渐进模态预对齐

- **摘录要点：** Stage 1 双向约束对齐 **动作 ↔ 潜空间世界动力学**；Stage 2 语义增强 + 多任务学习对齐 **动作 ↔ 视觉–语言**；Stage 3 在已对齐表征上 **VLM / 视频 / 机器人** 共训，固化「先预测未来、再生成动作」因果结构。
- **对 wiki 的映射：**
  - [Lumo-2](../../wiki/entities/lumo-2.md) — 训练管线 Mermaid 与 BAR 推理加速
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint 族 latent WAM 实例

### 2) 工程侧能力叙事

- **摘录要点：** **BAR** 块级自回归解码 **2.71×** 端到端加速（RTX 5090 + vLLM，93.53 ms vs 253.66 ms）；历史动作 **短程记忆** 缓解多阶段任务感知别名；原生支持 egocentric 人视频等异构数据微调。
- **对 wiki 的映射：**
  - [Lumo-2](../../wiki/entities/lumo-2.md) — 推理延迟与记忆机制
  - [Manipulation](../../wiki/tasks/manipulation.md) — 22 项真机评测任务索引

### 3) 评测与基线（项目页 + 技术报告一致）

- **摘录要点：** 具身 VLM 基准、可泛化 pick-and-place（Basic / Unseen Instructions / Unseen Objects）、**22 项** 真机挑战任务（时序推理 / 物理理解 / 控制复杂度）、人–机迁移（VisionPro + 多视角 egocentric 无动作标注视频）。
- **对 wiki 的映射：**
  - [Lumo-2](../../wiki/entities/lumo-2.md) — 任务表与视频–任务维度对照
  - [VLA](../../wiki/methods/vla.md) — 与 π₀.₅、Fast-WAM 同赛道对照语境

## 当前提炼状态

- [x] 项目页文案 + 全量演示视频 URL 已归档
- [x] 与技术报告 arXiv:2607.11270 交叉核对任务维度与三阶段训练
