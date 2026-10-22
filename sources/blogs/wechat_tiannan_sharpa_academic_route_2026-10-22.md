# Sharpa第一次披露学术路线，顺手开源最大的灵巧手触觉数据集

> 来源归档（blog / 微信公众号）

- **标题：** Sharpa第一次披露学术路线，顺手开源最大的灵巧手触觉数据集。
- **类型：** blog
- **作者：** 天南（公众号「天南具身公园」）
- **原始链接：** https://mp.weixin.qq.com/s/8QEV3jTXJWqcDQbIe5iDLw
- **发表日期：** 2026-10-22（微信页显示 Oct 22, 2026, 9:01 AM）
- **入库日期：** 2026-10-22
- **抓取方式：** `scripts/fetch_wechat_article.py`（容器 Chromium + 微信内置浏览器 UA），输出页面纯文本，不含图片。
- **原始落盘：** [`sources/raw/wechat_tiannan_sharpa_route_2026-10-22/article.md`](../raw/wechat_tiannan_sharpa_route_2026-10-22/article.md)
- **一句话说明：** 作者与 Sharpa 交流后，把 Sharpa 的灵巧操作学术线归纳为「数据从哪里来 / 动作如何落地 / 经验如何复用」三条路线，配合 CraftNet 三层架构与首个触觉数据集 SharpaDex（300 小时）。
- **项目页 / 源码：** 文内无链接；各项目开放状态见下表对应节点。

## 文中项目与本库节点映射

每个项目一个独立详情节点；同一项目已有节点的复用，不另建（CI 规则见 `schema/ingest-workflow.md` 步骤 3）。

| 项目（文内名） | 路线 | 本库节点 | 状态 |
|----------------|------|----------|------|
| SharpaDex（300 h 触觉数据集） | 数据 | [sharpadex](../../wiki/entities/sharpadex.md) | **本次新建** |
| IMCopilot + MoDE-VLA（削苹果） | 路线一 | [MoDE-VLA](../../wiki/entities/paper-sa-2603-08122-mode-vla-mixture-of-dexterous-experts-vla.md)（IMCopilot 是该论文的组件，同一项目） | 复用 |
| DexEMG | 路线一 | [paper-dexemg](../../wiki/entities/paper-dexemg.md) | 复用 |
| UVTA | 路线一 | [paper-uvta-human-tactile-dexterous-manipulation](../../wiki/entities/paper-uvta-human-tactile-dexterous-manipulation.md) | 复用 |
| Dex-X | 路线二 | [paper-dex-x](../../wiki/entities/paper-dex-x.md) | 复用 |
| GALATEA | 路线二 | [paper-galatea](../../wiki/entities/paper-galatea.md) | 复用 |
| WM-CraftNet | 路线二 | [paper-wm-craftnet](../../wiki/entities/paper-wm-craftnet.md) | 复用 |
| TacBPM | 路线二 | [paper-tacbpm](../../wiki/entities/paper-tacbpm.md) | 复用 |
| POISE | 路线二 | [paper-poise-in-hand-6d-pose-reaching](../../wiki/entities/paper-poise-in-hand-6d-pose-reaching.md) | **本次新建** |
| EgoScale | 路线三 | [EgoScale](../../wiki/entities/paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div.md) | 复用 |
| FTP-1 | 路线三 | [paper-ftp-1](../../wiki/entities/paper-ftp-1.md) | 复用 |
| CraftNet（三层 VTLA 架构） | 架构 | [craftnet](../../wiki/entities/craftnet.md) | **本次新建**（与 WM-CraftNet 是不同项目） |
| D01 整机 | 硬件 | [sharpa-d01-robot](../../wiki/entities/sharpa-d01-robot.md) | **本次新建** |
| W02 灵巧手 | 硬件 | [sharpa-w02-hand](../../wiki/entities/sharpa-w02-hand.md) | **本次新建** |
| AE01 触觉数据手套 | 硬件 | [sharpa-ae01-data-glove](../../wiki/entities/sharpa-ae01-data-glove.md) | **本次新建** |
| Sharpa Wave（22 DoF 手，一代） | 硬件 | 公司页 [sharpa-robotics](../../wiki/entities/sharpa-robotics.md) 中已有产品条目 | 复用 |

**仅作对照提及、不建节点：** Helix 02（时间线对比）、UMI、Manus 手套、NVIDIA SONIC（类比）、DQ（Dairy Queen）餐厅部署（见公司页）。

## 核心摘录（归纳，非全文）

- **三个数据断点**：遥操作感受不到机器手（手内操作如盘核桃几乎做不了）；人手数据（Ego / 手套）与机器手有 gap，姿态相同接触未必相同；多源数据（人手、灵巧手、夹爪、视触觉/阵列/力矩传感器）格式对不上。
- **两派 debate**：先把人类数据「翻译」成机器人可执行数据 vs 让模型一锅炖；文内称 Sharpa 两条都押。
- **CraftNet 时间线**：文内称 CraftNet 2026-01-07 公开、Helix 02 为 2026-01-27，早 20 天；System 2 约 1 Hz、System 1 内含约 2 Hz 世界模型与约 10 Hz 动作专家、System 0 约 100 Hz 触觉控制（文内表述；官方新闻稿只提 System 0/1，见 craftnet 节点）。
- **路线一**：IMCopilot 脚踏板让 RL 技能接管手内旋转（遥操作 31/90 → 80/90）→ DexEMG 肌电腕带（成功率 76% / 66% / 56%）→ UVTA 被动外骨骼直接采人（约 1000 条人类 + 150 条机器人示范，平均 70% 对最强基线 29%）。
- **路线二**：Dex-X（抓立方体 93%、桌面清扫 53%）与 GALATEA（1500+ 段参考轨迹）自上而下；WM-CraftNet（9 个物体预训练 → 49 个物体下游）居中；TacBPM 与 POISE（连续三目标 20% → 80%）自下而上，接口从「开关 → 轨迹 → 潜变量」。
- **路线三**：FTP-1 以触觉为锚学本体无关表征，约 3000 小时、26 来源、21 种触觉传感器；已见硬件微调平均 +17.2%，未见传感器仅训新编码器高出最强基线约 31.6 个百分点。FTP-1 可视为 EgoScale 之后的 middle train（文内表述）。
- **SharpaDex**：300 小时、至少 64.9% 时间处于密集接触、小指接触时间 < 5%；作者按 2000–4000 元/小时的草根调研价估算约 120 万元（作者估算，非官方）。
- **部署信号**：文内转述 Sharpa 向少卿称下一步将在餐饮、零售、酒店场景部署 3 万台机器人（转述，未见官方文件）。

## 参考来源

- [原始抓取](../raw/wechat_tiannan_sharpa_route_2026-10-22/article.md)
- 前一篇同主题文章：[机器之心：硬件、数据到模型](./wechat_sharpa_dexterous_manipulation_2026-10-16.md)
