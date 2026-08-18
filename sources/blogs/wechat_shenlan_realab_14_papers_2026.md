# 2026年：斯坦福宋舒然团队14篇工作全盘点

> 来源归档（blog / 微信公众号）

- **标题：** 2026年：斯坦福宋舒然团队14篇工作全盘点
- **类型：** blog
- **作者：** 深蓝AI（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/vcewu3wKIcrsidzfGr2-yg
- **发表日期：** 2026-08-18
- **入库日期：** 2026-08-18
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [`sources/raw/wechat_shenlan_realab_14_papers_2026-08-18/article.md`](../raw/wechat_shenlan_realab_14_papers_2026-08-18/article.md)
- **一句话说明：** 深蓝 AI 按三条技术脉络盘点 Stanford REALab（宋舒然）2026 年 14 篇代表性工作：基础模型与扩散策略微调、多模态感官融合与顺应控制、数据采集接口与跨具身操作；强调跨实体泛化、物理接触与野外数据采集。

## 核心摘录（归纳，非全文）

### 问题重框

- **2026 主线**：具身智能从单一「视觉–动作」映射，走向 **跨实体泛化** 与 **真实世界接触式操作**。
- **共同问题**：机器人物理形态千差万别、真实接触极其复杂时，如何打破硬件与数据壁垒，让智能体在更广阔空间自我进化？
- **三条脉络**（文内划分）：
  1. **机器人基础模型与策略微调** — 统一表征、扩散/生成式策略在线精炼、全栈迁移判断
  2. **多模态感官融合与顺应控制** — 力/触觉持续学习、免力传感器柔顺、UMI 野外力感知采集
  3. **数据采集接口与跨具身操作** — 模块化遥操作、单次示范泛化、全身移动操作、数据增广与 4D 视频、灵巧手功能重定向

### 文内收束判断

- REALab 2026 重点全面走向 **多模态融合、跨具身泛化、物理顺应性** 深水区。
- 未来具身智能需要更聪明的「大脑」（统一基础模型）、更具适应性的「小脑」（极简柔顺控制）、更灵活的数据获取方式（ModPack / UMI-FT / HoMMI 等）。

## 14 篇论文索引（标题以抓取版为准）

### 01 — 机器人基础模型与策略微调（5）

| # | 标题 | 会议/状态 | arXiv / 链接 | 开源结论（项目页核查，2026-08-18） |
|---|------|-----------|--------------|-----------------------------------|
| 01 | Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design | arXiv | [2607.25798](https://arxiv.org/abs/2607.25798) | **已开源** — [real-stanford/transformer-transformer](https://github.com/real-stanford/transformer-transformer) + ckpt |
| 02 | DF-ExpEnse: Diffusion Filtered Exploration for Sample Efficient Finetuning | ICML 2026 | [2606.19656](https://arxiv.org/abs/2606.19656) | **已开源** — [real-stanford/dfexpense](https://github.com/real-stanford/dfexpense) |
| 03 | From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning (DICE-RL) | ICML 2026 | [2603.10263](https://arxiv.org/abs/2603.10263) | **已开源** — [real-stanford/dice-rl](https://github.com/real-stanford/dice-rl) + HF 数据/ckpt |
| 04 | Are Foundation Models the Route to Full-Stack Transfer in Robotics? | 综述 | [2602.22001](https://arxiv.org/abs/2602.22001) | **无代码** — 迁移学习视角综述（Song 合著） |
| 05 | Gated Memory Policy: In-Context Memorization and Adaptation (GMP) | — | [2604.18933](https://arxiv.org/abs/2604.18933) | **已开源** — [gated-memory-policy.github.io](https://gated-memory-policy.github.io/) 列代码/数据/部署说明 |

### 02 — 多模态感官融合与顺应控制（3）

| # | 标题 | 会议/状态 | arXiv / 链接 | 开源结论 |
|---|------|-----------|--------------|----------|
| 06 | Multisensory Continual Learning: Adapting Pretrained Visuomotor Policies to Force (MuSe) | — | [2606.30988](https://arxiv.org/abs/2606.30988) | **部分** — 项目页 [jadenvc.github.io/multisensory-continual-learning](https://jadenvc.github.io/multisensory-continual-learning/) |
| 07 | Minimalist Compliance Control | RSS 2026 | [2603.00913](https://arxiv.org/abs/2603.00913) | **未列 GitHub** — 项目页 [minimalist-compliance-control.github.io](https://minimalist-compliance-control.github.io/)；算法无需学习、偏控制器实现 |
| 08 | In-the-Wild Compliant Manipulation with UMI-FT | ICRA 2026 | [2601.09988](https://arxiv.org/abs/2601.09988) | **已开源** — [real-stanford/UMI-FT](https://github.com/real-stanford/UMI-FT) 硬件+软件 |

### 03 — 数据采集接口与跨具身操作（6）

| # | 标题 | 会议/状态 | arXiv / 链接 | 开源结论 |
|---|------|-----------|--------------|----------|
| 09 | ModPack: An Extensible Teleoperation Interface for Bimanual Mobile Manipulation | — | [2607.19479](https://arxiv.org/abs/2607.19479) | **已开源** — [real-stanford/modpack](https://github.com/real-stanford/modpack) 硬件+遥操作软件（不含策略训练） |
| 10 | Behavior Prompting Policy: Demonstrations as Prompts for Manipulation (BPP) | — | [2606.30457](https://arxiv.org/abs/2606.30457) | **已开源** — [real-stanford/behavior_prompting](https://github.com/real-stanford/behavior_prompting) + iPhUMI |
| 11 | HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations | RSS 2026 | [2603.03243](https://arxiv.org/abs/2603.03243) | **已开源** — [github.com/xxm19/hommi](https://github.com/xxm19/hommi) 代码+数据+硬件 |
| 12 | One Demo Is Worth a Thousand Trajectories: Action-View Augmentation for Visuomotor Policies | CoRL 2025 | [2606.19586](https://arxiv.org/abs/2606.19586) | **部分** — [1001-demos.github.io](https://chuerpan.com/1001-demos.github.io/) |
| 13 | Geometry-Aware 4D Video Generation for Robot Manipulation | ICLR 2026 | [2507.01099](https://arxiv.org/abs/2507.01099) | **部分** — [robot4dgen.github.io](https://robot4dgen.github.io/) |
| 14 | DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation | ICML 2026 | [2505.24853](https://arxiv.org/abs/2505.24853) | **仿真 benchmark 已开** — [project-dexmachina.github.io](https://project-dexmachina.github.io/)；真机待检验 |

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [REALab 14 篇技术地图（2026）](../../wiki/overview/realab-14-papers-technology-map-2026.md) | **主沉淀页**：三条脉络阅读坐标 + 开源状态 + 本库交叉 |
| [海外具身智能实验室地图（2026）](../../wiki/overview/overseas-embodied-ai-labs-landscape-2026.md) | REAL Lab 节点补充 2026 论文簇 |
| [Diffusion Policy](../../wiki/methods/diffusion-policy.md) | #02 DF-ExpEnse、#03 DICE-RL、#05 GMP 等扩散策略微调/记忆线 |
| [Transformer Transformer（实体）](../../wiki/entities/paper-transformer-transformer.md) | #01 已深读 |
| [manipulation](../../wiki/tasks/manipulation.md)、[teleoperation](../../wiki/tasks/teleoperation.md) | UMI-FT / ModPack / HoMMI / BPP 数据采集与部署 |
| [cross-embodiment 选型](../../wiki/queries/cross-embodiment-transfer-strategy.md) | #01 机体共设计、#11 HoMMI 跨具身手眼 |

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 14 篇索引表 + 项目页开源核查
- [x] 升格 `wiki/overview/realab-14-papers-technology-map-2026.md`
- [ ] 单篇论文深读实体页（除 #01 Transformer Transformer 外，待后续逐篇 ingest）
