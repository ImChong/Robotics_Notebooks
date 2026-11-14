# 不改 VLA 主模型，怎样选出更可靠的动作？（2026-11-13，18 篇）

- **原文章：** https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w
- **来源：** 具身智能小站，2026-11-13（用户上传的 6 页 PDF）
- **内容：** 18 篇机器人研究论文的阅读优先级清单，不是原始实验论文。
- **核对范围：** 已完整阅读用户提供的 6 页转载 PDF；部分论文已直接打开 arXiv HTML 和官方项目/仓库。原论文与代码仍有待逐篇完整检查。
- **通读：** 已完成

## 阅读结构

原文建议 1 篇深读（REKI），2 篇跟进（Spark-0、HDH），其余 15 篇扫读。以下只保存索引，不把 18 篇论文内容合并成一个 wiki 详情页。

| 编号 | 论文 | arXiv | wiki | 入库状态 |
|---|---|---|---|---|
| 01 | REKI | [2610.22178](https://arxiv.org/abs/2610.22178) | [独立节点](../../wiki/entities/paper-reki-vla-register-memory.md) | 沿用已有节点 |
| 02 | Spark-0 | [2610.22342](https://arxiv.org/abs/2610.22342) | [独立节点](../../wiki/entities/paper-spark-0-spatial-tactile-pretraining.md) | 沿用已有节点 |
| 03 | HDH | [2611.14834](https://arxiv.org/abs/2611.14834) | [独立节点](../../wiki/entities/paper-hdh-action-verifier.md) | 本次新建草稿 |
| 04 | CarryOn | [2610.21453](https://arxiv.org/abs/2610.21453) | [独立节点](../../wiki/entities/paper-carryon-dual-loop.md) | 本次新建草稿 |
| 05 | PlayPolicy | [2610.22340](https://arxiv.org/abs/2610.22340) | [独立节点](../../wiki/entities/paper-dsp-supervised-play.md) | 沿用已有节点 |
| 06 | FastVLA | [2610.22146](https://arxiv.org/abs/2610.22146) | [独立节点](../../wiki/entities/paper-fastvla-task-adaptive-token-compression.md) | 沿用已有节点 |
| 07 | Mem4Act | [2610.22323](https://arxiv.org/abs/2610.22323) | [独立节点](../../wiki/entities/paper-mem4act-latent-memory.md) | 沿用已有节点 |
| 08 | Sub-Millisecond Reflexes | [2610.21913](https://arxiv.org/abs/2610.21913) | [独立节点](../../wiki/entities/paper-submillisecond-reflex-engine.md) | 本次新建草稿 |
| 09 | DemoStream | [2610.22020](https://arxiv.org/abs/2610.22020) | [独立节点](../../wiki/entities/paper-demostream-batched-tamp.md) | 本次新建草稿 |
| 10 | TRACE | [2610.21515](https://arxiv.org/abs/2610.21515) | [独立节点](../../wiki/entities/paper-trace-conversational-humanoid-motion.md) | 本次新建草稿 |
| 11 | Compliant GMT | [2610.22119](https://arxiv.org/abs/2610.22119) | [独立节点](../../wiki/entities/paper-compliant-gmt.md) | 沿用已有节点 |
| 12 | CoDA | [2610.22362](https://arxiv.org/abs/2610.22362) | [独立节点](../../wiki/entities/paper-coda-coevolving-data-policy.md) | 本次新建草稿 |
| 13 | Whole-Body Dexterity | [2610.22184](https://arxiv.org/abs/2610.22184) | [独立节点](../../wiki/entities/paper-whole-body-dexterity.md) | 沿用已有节点 |
| 14 | PRM4RL | [2610.22056](https://arxiv.org/abs/2610.22056) | [独立节点](../../wiki/entities/paper-prm4rl-progress-reward.md) | 本次新建草稿 |
| 15 | EM-ViT | [2611.14958](https://arxiv.org/abs/2611.14958) | [独立节点](../../wiki/entities/paper-emvit-online-metric-depth.md) | 本次新建草稿 |
| 16 | MOSS | [2610.22193](https://arxiv.org/abs/2610.22193) | [独立节点](../../wiki/entities/paper-moss-mode-source-shaping.md) | 本次新建草稿 |
| 17 | SPEAR | [2610.21621](https://arxiv.org/abs/2610.21621) | [独立节点](../../wiki/entities/paper-spear-selective-preference.md) | 本次新建草稿 |
| 18 | Target-Aware Language Map | [2610.21619](https://arxiv.org/abs/2610.21619) | [独立节点](../../wiki/entities/paper-target-aware-gaussian-langmap.md) | 本次新建草稿 |

## 重要辨析

- REKI 将短期历史压入动作专家寄存器；HDH 保持 VLA 权重不变，用轻量验证器从动作候选中择优，两者不是同一种机制。
- Spark-0 的训练期触觉监督不能混同于纯视觉零样本迁移。
- 亚毫秒反射论文报告的低于 0.1ms 指控制器切换，不是机器人接触后完整卸力时间。
- CoDA 项目页标注 Code coming soon；EM-ViT 与 MOSS 项目页的代码链接是占位地址；不能仅凭项目页存在就认定源码可运行。

## 后续核查清单

- [ ] 逐篇完成 arXiv 正文及附录通读；将对应 sources/papers 标记为已完成
- [ ] 对已开源项目阅读真实训练、推理入口及源码调用链，补充运行时序 Mermaid 图
- [ ] 运行仓库 ci-preflight 与 health report，消除剩余告警后将草稿转为正式 PR
