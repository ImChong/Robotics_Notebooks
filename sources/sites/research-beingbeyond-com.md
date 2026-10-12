# 智在无界（BeingBeyond）Research 站核查

> 来源归档（site / BeingBeyond 官方）

- **URL：** <https://research.beingbeyond.com/>（公司官网 <https://beingbeyond.com>；GitHub <https://github.com/BeingBeyond>；Hugging Face <https://huggingface.co/BeingBeyond>；X <https://x.com/beingbeyond_>）
- **机构：** 北京智在无界科技有限公司（官网页脚）；英文名 BeingBeyond；地址「北京市海淀区鼎好DH3A座702室」
- **核查日期：** 2026-10-12
- **抓取方式：** Research 站是 Next.js 客户端渲染，正文显示「Loading...」；用 `curl` 读取首页 HTML 中的 RSC 数据，取出 26 条研究的 slug、标题、TL;DR、官网日期与会议；再逐个读取 `/<slug>` 子页提取 arXiv / GitHub / HF 链接。arXiv v1 日期用 `export.arxiv.org/api/query` 批量核对。GitHub 组织 API 在本环境不可用，改读组织仓库列表网页；HF 用公开 models / datasets API。
- **覆盖 wiki：** [智在无界（BeingBeyond）](../../wiki/entities/beingbeyond.md)
- **站点自述：** 「BeingBeyond is a startup company on foundation models for embodied intelligence.」；页脚「© 2025-2026」「Embody the future」

## 公司官网（beingbeyond.com）摘录

- 简介：「我们是一家专注于人工智能技术研发的创新企业，致力于解决具身智能的核心技术难题，引领人形机器人技术革命」
- 使命 / 愿景：「以第一性原理驱动创新」/「引领物理智能的未来」
- 模型线：Being-H（灵巧操作手部动作生成）、Being-M（全身动作生成）、Being-VL（跨模态基础模型）、Being-W（全身运动控制）
- 解决方案：D1（「首个桌面级灵巧手机械臂」）、Being-Dex（机器人通用灵巧操作解决方案）、Being-Actor（人形机器人全身摇操系统）、U1（「全球首款 Real DexUMI」）
- 新闻 3 条均无日期：U1 发布、Being-Dex、BumbleBee。图片路径分别含 `2026/03/18`、`2026/01/19`、`2026/01/19`，只能推测为上传时间（推测）。
- 官网未写成立日期与创始人。

## 成立与融资（媒体口径）

| 来源 | 日期 | 要点 |
|------|------|------|
| [凤凰网科技（转硬氪）](https://tech.ifeng.com/c/8kApGyTicks) | 2025-06-14 | 「成立于2025年1月」；创始人卢宗青为北京大学计算机学院长聘副教授，曾任智源研究院多模态交互研究中心负责人；多名核心成员来自智源；数千万元融资，联想之星领投，智谱 Z 基金、燕缘创投、彬复资本跟投 |
| [36 氪 / 智能涌现](https://eu.36kr.com/zh/p/3653424523682183) | 2026-01-24 | 「于2025年创立」，「正式开始运营是在2025年5月」；天使轮数千万元，考拉基金（拉卡拉旗下）领投，领航新界、灵心巧手跟投，联想之星、星连资本加注；Being-H0.5 可跑在 Orin-NX 等端侧芯片；Being-Dex 为付费后训练服务；客户含 PND（Adam-U Ultra）与灵心巧手 |

两篇报道对轮次名称写法不同（前者未写轮次，后者称天使轮），本页不推定两轮是否为同一轮。

## Research 列表（2026-10-12 共 26 条）

官网日期取首页卡片 `dateLabel`；arXiv v1 来自 arXiv API。两者月份全部一致，日期差 1–16 天。

| 官网日期 | slug | 标题（简） | 官网标注会议 | arXiv（v1 日期） | 官网代码链接 |
|----------|------|-----------|--------------|------------------|--------------|
| 2024-10-03 | being-vl0 | Being-VL-0: From Pixels to Tokens（BPE 视觉 tokenizer） | ICLR 2025 | [2410.02155](https://arxiv.org/abs/2410.02155)（2024-10-03） | BeingBeyond/Being-VL-0 |
| 2024-10-03 | being-m0 | Being-M0: Scaling Motion Generation Models with Million-Level Human Motions | ICML 2025 | [2410.03311](https://arxiv.org/abs/2410.03311)（2024-10-04） | BeingBeyond/Being-M0 |
| 2025-03-11 | being-0 | Being-0: A Humanoid Robotic Agent with VLMs and Modular Skills | — | [2503.12533](https://arxiv.org/abs/2503.12533)（2025-03-16） | BeingBeyond/Being-0 |
| 2025-06-20 | rlpf | RLPF: RL from Physical Feedback for Humanoid Control | ECCV 2026 | [2506.12769](https://arxiv.org/abs/2506.12769)（2025-06-15） | BeingBeyond/RLPF |
| 2025-06-20 | bumblebee | BumbleBee: From Experts to a Generalist | NeurIPS 2025 Spotlight | [2506.12779](https://arxiv.org/abs/2506.12779)（2025-06-15） | BeingBeyond/BumbleBee |
| 2025-06-26 | being-vl05 | Being-VL-0.5: Unified Multimodal Understanding via Byte-Pair Visual Encoding | ICCV 2025 Highlight | [2506.23639](https://arxiv.org/abs/2506.23639)（2025-06-30） | BeingBeyond/Being-VL-0.5 |
| 2025-07-21 | being-h0 | Being-H0: VLA Pretraining from Large-Scale Human Videos | ICML 2026 | [2507.15597](https://arxiv.org/abs/2507.15597)（2025-07-21） | BeingBeyond/Being-H0；HF 权重 |
| 2025-08-01 | being-m05 | Being-M0.5: A Real-Time Controllable Vision-Language-Motion Model | ICCV 2025 | [2508.07863](https://arxiv.org/abs/2508.07863)（2025-08-11） | BeingBeyond/Being-M0.5 |
| 2025-09-22 | demograsp | DemoGrasp: Universal Dexterous Grasping from a Single Demonstration | ICLR 2026 | [2509.22149](https://arxiv.org/abs/2509.22149)（2025-09-26） | BeingBeyond/DemoGrasp |
| 2025-10-13 | demohlm | DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation | RAL | [2510.11258](https://arxiv.org/abs/2510.11258)（2025-10-13） | BeingBeyond/DemoHLM |
| 2025-12-01 | dig | DiG: Transport Discrepancy as a Reliability Signal for VLA Models | ECCV 2026 | [2512.01715](https://arxiv.org/abs/2512.01715)（2025-12-01） | BeingBeyond/DiG |
| 2025-12-15 | demofungrasp | DemoFunGrasp: Universal Dexterous Functional Grasping via Demonstration-Editing RL | CVPR 2026 | [2512.13380](https://arxiv.org/abs/2512.13380)（2025-12-15） | BeingBeyond/DemoFunGrasp |
| 2025-12-15 | vipa-vla | VIPA-VLA: Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos | CVPR 2026 | [2512.13080](https://arxiv.org/abs/2512.13080)（2025-12-15） | BeingBeyond/VIPA-VLA |
| 2025-12-24 | unitachand | UniTacHand: Unified Spatio-Tactile Representation for Human to Robotic Hand Skill Transfer | — | [2512.21233](https://arxiv.org/abs/2512.21233)（2025-12-24） | BeingBeyond/UniTacHand |
| 2026-01-20 | being-h05 | Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization | CoRL 2026 Oral | [2601.12993](https://arxiv.org/abs/2601.12993)（2026-01-19） | BeingBeyond/Being-H；HF collection |
| 2026-02-09 | rethink_vla | Rethinking VLA Model Scaling: Alignment, Mixture, and Regularization | — | [2602.09722](https://arxiv.org/abs/2602.09722)（2026-02-10） | BeingBeyond/Rethink_VLA |
| 2026-02-12 | fast | FAST: General Humanoid Whole-Body Control via Pretraining and Fast Adaptation | CoRL 2026 | [2602.11929](https://arxiv.org/abs/2602.11929)（2026-02-12） | BeingBeyond/FAST |
| 2026-02-26 | jala | JALA: Joint-Aligned Latent Action for Scalable VLA Pretraining in the Wild | CVPR 2026 | [2602.21736](https://arxiv.org/abs/2602.21736)（2026-02-25） | BeingBeyond/JALA |
| 2026-03-17 | ptr | PTR: Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting | — | [2603.16542](https://arxiv.org/abs/2603.16542)（2026-03-17） | BeingBeyond/PTR |
| 2026-03-19 | opent2m | OpenT2M: No-frill Motion Generation with Open-source, Large-scale, High-quality Data | CVPR 2026 | [2603.18623](https://arxiv.org/abs/2603.18623)（2026-03-19） | 子页未列 |
| 2026-04-14 | being-h07 | Being-H0.7: A Latent World-Action Model from Egocentric Videos | NeurIPS 2026 | [2605.00078](https://arxiv.org/abs/2605.00078)（2026-04-30） | BeingBeyond/Being-H |
| 2026-04-20 | better | BeTTER: Unmasking the Illusion of Embodied Reasoning in VLA Models | ECCV 2026 | [2604.18000](https://arxiv.org/abs/2604.18000)（2026-04-20） | BeingBeyond/BeTTER |
| 2026-06-04 | realdexumi | RealDexUMI: A Wearable Universal Manipulation Interface for Dexterous Robot Learning | CoRL 2026 | [2606.06033](https://arxiv.org/abs/2606.06033)（2026-06-04） | 子页未列 |
| 2026-07-02 | ttp | TTP: Human-Centric Transferable Tactile Pre-Training for Dexterous Manipulation | — | [2607.01067](https://arxiv.org/abs/2607.01067)（2026-07-01） | BeingBeyond/TTP；HF 权重与数据 |
| 2026-07-14 | being-m07 | Being-M0.7: A Latent World-Action Model for Humanoid Robots | — | 无（BeingBeyond Technical Report） | 子页未列 |
| 2026-07-28 | being-h08 | Being-H0.8: A Latent Tactile World-Action Model at Scale | — | 无（BeingBeyond Technical Report） | 子页未列 |

- Being-H0.7 官网日期 2026-04-14 早于 arXiv v1（2026-04-30）16 天，月份相同。
- 2024 年的 Being-VL-0、Being-M0 早于公司成立（2025-01，媒体口径），由官网 Research 页收录；本页不推定其为公司成立后的成果。
- 所有 arXiv 论文末位作者均为 Zongqing Lu（卢宗青）；Rethink VLA 末位作者为 Qin Jin。

## Being-H0.8 项目页摘录（无 arXiv）

- 引用格式：`howpublished={BeingBeyond Technical Report}`，作者 `BeingBeyond Team`，2026。
- 自述「latest general-purpose embodied foundation model」；把 Being-H0.7 的潜空间 World–Action Model 从视觉预测扩展到触觉感知交互。
- 数据：UniHand 3.0，500,000+ 小时第一视角人类视频（公开集合 + 数据合作方 + 自建），称每条样本可追溯来源；另有合成人–机对齐数据与任务对齐的人类多模态示范。
- TactoHand：从人手–物体交互视频推断 MANO 表面上的接触与接近度两种稠密场，作为伪触觉监督。
- TopoHand：基于骨架的统一状态–动作空间，替代 Being-H0.5 的第一代 Unified Action Space；覆盖人手、灵巧手与平行夹爪。
- 训练沿用 Being-H0.7 的先验–后验结构；新增 slow–fast 动作专家（慢流缓存视觉语言与潜世界上下文，快流用最新本体与触觉反馈重生成下一段动作）与通用触觉编码器（Perceiver 重采样，缺触觉时用可学习占位）。
- 真机：两台 ROKAE AR5 双臂，换装 LinkerHand L25（16 主动 / 21 总自由度，指尖压阻阵列）与 DexHand 021；演示挤软管、袋中摸取、毛笔书写、擦污渍、不同硬度水果抓取等。
- 项目页未列代码、权重或数据下载入口（2026-10-12）。

## 开源渠道核查（2026-10-12）

| 渠道 | 结果 | 方法 |
|------|------|------|
| GitHub 组织 `BeingBeyond` | 27 个公开仓库；Being-H 约 1.2k star。仓库含论文代码、D1 SDK（Beingbeyond_D1 / D1_edu）及 OpenMMEgo、MEgoHand、Jaeger 等官网未列项目 | 组织仓库列表网页 |
| Hugging Face `BeingBeyond` | 13 个模型（Being-H0 1B / 8B / 14B、Being-H05-2B 及 LIBERO / RoboCasa 微调、TTP、BeingVL tokenizer 等）；14 个数据集（H0 后训练、PND Adam-U 任务、UniHand_Preview、H-Tac_Sample、Being-H-EDU_SO101 等） | `huggingface.co/api/models?author=BeingBeyond`、`/api/datasets?author=BeingBeyond` |
| Being-M0.7 / Being-H0.8 | 项目页只有 PDF / 演示，未列代码、权重 | 子页 HTML |

仓库存在不等于训练代码、权重与数据完整开放；复现前逐仓核对 README。

## 本库既有标注更正（2026-10-12）

依据官网页脚「北京智在无界科技有限公司」与英文名 BeingBeyond：

- `schema/institutions.json`：`beingbeyond` 标签由「超越智能（BeingBeyond）」改为「智在无界（BeingBeyond）」；删除无其他引用的 `agilex-ai`（「智在无界（AgileX AI）」），其别名 `zhizaiwujie` 并入 `beingbeyond`。
- `wiki/entities/cn-os-*` 中 12 个 BeingBeyond 仓库页的 `institutions` 由 `agilex-ai` 改为 `beingbeyond`；生成脚本 `scripts/patch_cn_os_institutions.py` 的「智在无界」映射同步改为 `beingbeyond`。
- DemoHLM 详情与 3 个来源归档、[Being-M0.7 论文归档](../papers/being_m07.md) 中的「超越智能」改为「智在无界」。
- Being-H 仓库分类由「数据集/Benchmark」（国内开源全景原清单）改为「VLA/操作模型」：GitHub 描述为 Human-centric embodied foundation model family，Research 站 Being-H0.5 / H0.7 的代码链接均指向该仓库；同步仓库页、来源归档、424 覆盖索引与映射 JSON，原文归档不改。
