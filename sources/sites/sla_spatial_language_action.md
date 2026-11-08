# SLA 官方项目主页与资源核查（2026-11-08）

- **项目：** SLA: Scaling Spatial-Language-Action Model via Continual Pre-training / 通过持续预训练扩展空间-语言-动作模型
- **论文：** https://arxiv.org/abs/2611.08373 （v1，2026-11-05）
- **论文全文：** https://arxiv.org/html/2611.08373v1
- **已访问的官方项目主页：** https://sla-team.github.io/omnisf_projectpage/
- **已访问的另一官方入口：** https://embodiedsla.github.io/ （作者在 GitHub / HF 论文页提供，展示同一 SLA 项目内容）
- **代码：** https://github.com/OpenHelix-Team/Spatial-Forcing/tree/master/SLA
- **Hugging Face 论文页：** https://huggingface.co/papers/2611.08373
- **作者模型/数据资源合集：** https://huggingface.co/collections/haofuly/sla-collection
- **核查范围：** 实际打开上述两个项目主页、Hugging Face 论文页与作者 Collection，核对正文、性能口径、资源条目；对照官方 GitHub SLA/README 的训练/评测状态与论文 HTML v1。资源页面显示可访问，不等于本次已下载权重或完成复现。
- **开放状态：** 部分开源。已有可访问代码、训练组件、模型和部分评测检查点；作者未公开完整持续预训练原始数据及 RoboDojo-Real 完整训练/评估实现。

## 项目页直接验证的内容

两个作者提供的项目入口目前均能读取相同研究正文，包含署名及机构、方法概览、六来源数据表、四个指标卡片和真实部署说明。此前归档中「项目主页无法取得正文」的备注已过期。

- **框架：** 基于 π0.5 持续预训练，使用冻结 3D 模型约束 VLA 中间视觉 token；空间教师和对齐分支仅参与训练，常规推理结构保持原有策略。
- **六来源原始统计：** 约 171.39 万 episodes / 14,330 小时，属于数据源的合计，**不是**清洗后的实际预训练样本量。
- **页面报告指标（均为论文作者实验）：** LIBERO-Plus 87.4%（按七类扰动回合数加权）、RoboTwin 2.0 Clean2Random 72.3%（50 任务）、RoboCasa-Panda 68.4%（24 任务）、RoboDojo-Real 33.9%（三种本体整体成功率）。
- **8 组真实部署条件：** 平均成功率相对 π0.5 提高 24.0 个百分点；论文 Table 11 的 SLA 83.8%、π0.5 59.8% 与网页叙述一致。此项并非上述四项基准的平均。

## Hugging Face 资源逐项核对

| 官方资源 | 直接链接 | 本次确认的角色与边界 |
|---|---|---|
| SLA_base | https://huggingface.co/haofuly/SLA_base | 模型仓；README 用作持续预训练后的基座权重下载 |
| SLA_training_component | https://huggingface.co/haofuly/SLA_training_component | 模型仓；README 用作 OmniVGGT、DINOv2 等训练辅助组件 |
| SLA_benchmark_ckpts | https://huggingface.co/haofuly/SLA_benchmark_ckpts | 模型仓；README 用于下游评测的后训练权重 |
| SLA_real_world_demo | https://huggingface.co/datasets/haofuly/SLA_real_world_demo | Dataset 仓；真实场景演示资源，不等同完整持续预训练数据集 |

作者 Collection 当前列出以上四项；HF 论文页用于索引、讨论与链接，不是新的方法论文或可下载权重。以上权重仓部分未提供模型卡或完整说明，应以官方 README 的路径和兼容配置为准。

## 开源边界与复现入口

官方 SLA/README 的 TODO 表中 LIBERO-Plus、RoboTwin 后训练和评估、continual pre-training corpus 实现项为已完成，**但另列的 continual pre-training data 仍标记未开放**；RoboDojo-Real 后训练/评估也未完成公开。不能将「代码项已实现」等同「全量训练数据已提供」。

- [官方训练与评估说明](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/README.md)
- [作者资源合集](https://huggingface.co/collections/haofuly/sla-collection)
- [SLA Wiki 实体页](../../wiki/entities/paper-sla-spatial-language-action.md)
- [论文原文归档](../papers/sla_spatial_language_action_arxiv_2611_08373.md)
- [官方代码归档](../repos/sla_openhelix.md)
