# Δ-flow 项目页 / Project Website

> 来源归档（site）｜一手项目页核查，2026-11-07

- **标题：** Δ Flow Matching for Robot Foundation Models
- **类型：** site
- **链接：** <https://zuo-kuangji.github.io/delta-flow/>
- **论文：** <https://arxiv.org/abs/2610.15794>
- **官方仓库：** <https://github.com/NTUMARS/delta-flow-matching>
- **机构：** 南洋理工大学 MARS Lab
- **首次入库：** 2026-10-14
- **本次核验：** 2026-11-07
- **沉淀到 wiki：** [Δ-flow 主实体](../../wiki/entities/paper-delta-flow-matching-robot-foundation-models.md)

## 核验事实

- 项目页包含方法示意、ImageWAM / Fast-WAM / π0.5 三类架构、LIBERO 与 LIBERO-Plus 结果、训练和延迟图及消融。
- 主页宣称 ImageWAM 达到 90% LIBERO 成功率所需 epoch 从 6 到 1；相对十步高斯，三种架构一步推理加速 1.54–4.45 倍。该口径已由 [arXiv v1](../papers/delta_flow_matching_robot_foundation_models_arxiv_2610_15794.md) 交叉核验。
- 页面部分 “arXiv soon / PDF soon” 或互动演示为旧文案；论文已于 2026-10-11 上线。页面互动曲线明确属于 illustration，不能当成实测数据。
- 页面链接官方 NTUMARS 仓库；仓库仅有 README、示意资源，没有公开可运行训练/评估实现、权重。

## 开源状态

**待发布（仅论文、项目页和官方仓库说明公开）**。不要仅凭 GitHub 入口可访问就写「已开源」。待训练/评测源码发布后再复核。
