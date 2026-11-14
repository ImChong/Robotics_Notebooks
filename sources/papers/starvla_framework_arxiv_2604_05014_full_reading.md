# StarVLA 框架原文核对 / A Lego-like Codebase for Vision-Language-Action Model Developing

- **类型：** paper / 框架论文
- **主论文：** https://arxiv.org/html/2604.05014v1（2026-04-06）
- **项目页：** https://starvla.github.io/
- **代码：** https://github.com/starVLA/starVLA （默认分支 `starVLA_dev`）
- **机构：** StarVLA Community、香港科技大学冯诺依曼研究院（Von Neumann Institute, HKUST）
- **身份边界：** 与 **StarVLA-α: Reducing Complexity in Vision-Language-Action Systems**（arXiv:2604.11757）是两篇不同论文；本页只对应 **框架论文 2604.05014**，后者在现有 `wiki/methods/star-vla.md` 和 `sources/papers/star_vla.md` 介绍。
- **核对范围：** 2026-11-14 核对 arXiv HTML v1 §1–8、作者与基准附表（统一模型与数据接口、训练/评测/部署、LIBERO、SimplerEnv、RoboCasa、RoboTwin、双模态共训、多基准与训练效率）；官方 `starVLA_dev` README 全部章节与 `docs/starVLA_guideline.md` 的安装、数据、训练与 server/client 流程。
- **通读：** 已完成

## 方法与结果边界

- 核心设计是可替换的 **VLM/世界模型 backbone + action head**，共享数据结构、训练器与评测接口。
- action head 包含 FAST 自回归、OFT 并行回归、π 流匹配、GR00T 双系统四种实例；README 还支持 Wan/Cosmos 等世界骨干和异构 co-training。
- 论文按标准 LIBERO (10 tasks × 50 episodes/suite)、RoboTwin、RoboCasa、SimplerEnv 组织实验；一个 checkpoint 不自动跨所有 benchmark 零样本有效。
- 官方 README 的训练 Recipe 列有 SFT、多模态联合和跨本体 co-train；**RL adaptation 仍显示待办**，不得写成成熟全路径。
- 训练效率实验依赖 A100 数量及 DeepSpeed 配置，不应将吞吐数字当成独立动作推理速度。
- **代码流程：** `examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh` → `accelerate launch` + `starVLA/config/deepseeds/deepspeed_zero2.yaml` → 加载 `starvla_cotrain_libero.yaml`、dataloader 与选择的 framework。评测由 policy server 和独立 LIBERO simulator client 组成；均依据官方 Quick Start Guide，未进行 GPU 运行。

## 对 wiki 的映射

- [StarVLA 框架唯一论文详情](../../wiki/entities/paper-starvla-framework.md)
- [独立 StarVLA-α 方法页](../../wiki/methods/star-vla.md)
