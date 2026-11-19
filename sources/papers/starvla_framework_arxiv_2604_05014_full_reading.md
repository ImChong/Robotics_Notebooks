# StarVLA 框架原文核对 / A Lego-like Codebase for Vision-Language-Action Model Developing

- **类型：** paper / 框架论文
- **主论文：** https://arxiv.org/html/2604.05014v1（2026-04-06）
- **项目页：** https://starvla.github.io/
- **代码：** https://github.com/starVLA/starVLA （默认分支 `starVLA_dev`）
- **机构：** StarVLA Community、香港科技大学冯诺依曼研究院（Von Neumann Institute, HKUST）
- **身份边界：** 与 **StarVLA-α: Reducing Complexity in Vision-Language-Action Systems**（arXiv:2604.11757）是两篇不同论文；本页只对应 **框架论文 2604.05014**，后者在 `wiki/entities/paper-starvla-alpha.md` 与 `sources/papers/star_vla.md` 介绍，方法页保留综合入口。
- **核对范围：** 2026-11-14 核对 arXiv HTML v1 §1–8、作者与基准附表（统一模型与数据接口、训练/评测/部署、LIBERO、SimplerEnv、RoboCasa、RoboTwin、双模态共训、多基准与训练效率）；官方 `starVLA_dev` README 全部章节与 `docs/starVLA_guideline.md` 的安装、数据、训练与 server/client 流程。
- **通读：** 已完成
- **2026-11-19 增量核对：** 复核 arXiv HTML v1 §2.2–3.2、§5.3 Table 5/6、§7.1–7.2 Table 9，官方项目页及 2026-11-19 的 GitHub `starVLA_dev` README、`docs/starVLA_guideline.md`、LIBERO YAML/启动脚本与 `QwenOFT.py` 的 `forward`/`predict_action` 调用。既有 2026-11-14 主论文通读记录保留。

## 方法与结果边界

- 核心设计是可替换的 **VLM/世界模型 backbone + action head**，共享数据结构、训练器与评测接口。
- action head 包含 FAST 自回归、OFT 并行回归、π 流匹配、GR00T 双系统四种实例；README 还支持 Wan/Cosmos 等世界骨干和异构 co-training。
- 论文按标准 LIBERO (10 tasks × 50 episodes/suite)、RoboTwin、RoboCasa、SimplerEnv 组织实验；一个 checkpoint 不自动跨所有 benchmark 零样本有效。
- 官方 README 的训练 Recipe 列有 SFT、多模态联合和跨本体 co-train；**RL adaptation 仍显示待办**，不得写成成熟全路径。
- 训练效率实验依赖 A100 数量及 DeepSpeed 配置，不应将吞吐数字当成独立动作推理速度。
- **代码流程：** `examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh` → `accelerate launch` + `starVLA/config/deepseeds/deepspeed_zero2.yaml` → 加载 `starvla_cotrain_libero.yaml`、dataloader 与选择的 framework。评测由 policy server 和独立 LIBERO simulator client 组成；均依据官方 Quick Start Guide，未进行 GPU 运行。

## 对 wiki 的映射

- [StarVLA 框架唯一论文详情](../../wiki/entities/paper-starvla-framework.md)
- [独立 StarVLA-α 论文详情](../../wiki/entities/paper-starvla-alpha.md)
- [StarVLA 方法入口](../../wiki/methods/star-vla.md)

## 2026-11-19 动作头消融、联合训练与复现口径补充

- **单基准 RoboCasa-GR1（§5.3、Table 5）**：24 个 Tabletop 任务、每任务 50 rollouts；模型在本基准官方训练数据上训练，训练使用 16×A100、最多 100K steps。FAST **39.0%**、π **43.9%**、GR00T **47.8%**、OFT **48.8%**。OFT 与 FAST 相差 **9.8 个百分点**，指特定训练/测试协议下的成功率差，并非相对提升。
- **同论文不同表的口径问题**：Table 5 OFT **48.8%**；Table 9 Specialist OFT 行 **53.8%**，§7.2 正文却引用单基准最好值 **48.8%**。无法仅据原文把差异归因到某一变量。跨表比较时分别写明原表；Table 9 的 Generalist 57.3% 对同表 Specialist OFT 53.8% 为 **3.5 个百分点**。
- **联合四基准（§7、Table 9）**：LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1 的训练数据混合，单模型训练；学习率 1e-4、总 batch 256，异构低维动作用 32 维 padding；按各基准原官方协议评测，未做 benchmark-specific 二次微调。RoboCasa-GR1 **57.3%**；它**见过该基准训练数据**，不可称未见域零样本。
- **源码对应关系（2026-11-19 官方开发分支静态核对）**：`examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml` 的 `framework.name: QwenGR00T`；`run_libero_train.sh` 将 `Framework_name=QwenPI` 和 `base_vlm=.../Qwen3.5-0.8B` 覆盖为 CLI 参数，经 `accelerate launch` 调 `starVLA/training/train_starvla.py`；Quick Start 当前给出 `QwenOFT`、`QwenFAST`、`QwenPI_v3`、`QwenGR00T` 四个名字。必须以最终运行配置/分支为准，不能把启动脚本当前值误写为论文对照实验设置。
- **OFT 动作解码代码**：`starVLA/model/framework/VLM4A/QwenOFT.py` 的 `predict_action` 抽取动作 token hidden states 并调用 `action_model.predict_action`；训练 `forward` 计算 `masked_l1_loss`，与论文 §2.3 对应。仅静态阅读，不代表已实际运行。
- **RL 开放状态时间线**：arXiv v1（2026-04-06）§3.1.4 将 RLinf 联动写为正在集成；官方 README 日期为 2026-04-09 的更新已列 [StarVLA × RLinf 教程](https://rlinf.readthedocs.io/)，不能笼统写成 2026-11 仍完全待办，也不能由单一外部教程推导所有 RL 算法已支持。
- **代码开源**：仓库及官方项目主页已公开；官方 README 说明 `starVLA_dev` 为活跃开发/可能不稳定的分支，已验证结果应核对 `starVLA` 稳定分支；FAST 的预训练 backbone 还须使用动作词表匹配版本。本文成功率取自作者报告，未复现实验。

## 原始资料与交叉引用

- [论文 HTML v1](https://arxiv.org/html/2604.05014v1)（§5.3、§7、Table 5/6/9）
- [StarVLA 官方主页](https://starvla.github.io/) · [官方论文页](https://starvla.github.io/papers/starvla/)
- [官方 GitHub README](https://github.com/starVLA/starVLA) · [Quick Start](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/starVLA_guideline.md)
- [LIBERO 启动脚本](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh) · [YAML 配置](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml) · [OFT 源码](https://github.com/starVLA/starVLA/blob/starVLA_dev/starVLA/model/framework/VLM4A/QwenOFT.py)
- [官方项目页归档](../sites/starvla-framework-official-project.md) · [GitHub 仓库归档](../repos/star_vla.md)

