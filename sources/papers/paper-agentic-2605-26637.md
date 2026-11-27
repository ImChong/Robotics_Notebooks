# Enabling Extensible Embodied Capabilities with Tools

- **作者：** Xueyang Zhou, Zijia Wang, Qianjiang Li, Yibo Hu, Guiyao Tie, Li Wan, Yidan Liu, Pan Zhou*, Lichao Sun, Yongchao Chen*（* Corresponding authors: Pan Zhou, Yongchao Chen）
- **机构：** Huazhong University of Science and Technology (HUST), Hebei University of Technology, Tianjin University, Lehigh University, College of AI, Tsinghua University
- **版本 / 载体：** arXiv:2605.26637v1 [cs.RO]（2026-05-26 提交，51 页，20 张图，CC BY 4.0），arXiv HTML 全文
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2605.26637) / [arXiv 摘要页](https://arxiv.org/abs/2605.26637) / [项目主页 Manip Tool Hub](https://racingemperor.github.io/manip-tool-hub/)
- **项目页 / 代码：** 未确认开源代码 — 项目主页 [`https://racingemperor.github.io/manip-tool-hub/`](https://racingemperor.github.io/manip-tool-hub/)（HTTP 200 可访问）展示了工具目录、能力分类与评测快照；论文正文给出的 GitHub 仓库地址 `https://github.com/Zxy-MLlab/EmbodiedTool` 截至 2026-11-27 核查尚不存在（`exists: false`），暂未确认可运行官方源码实现
- **核对范围：** 通读全文 §1–§6、Table 1–4、Figure 1–5、References [1]–[55] 及全部附录 Appendix A–I（含 D.1–D.2 工具分类学与收集流水线、Table 5 代表性工具目录 T01–T22、E.1–E.4 EmbodiedToolBench 四项任务数据构造原则、F.1–F.2 112 个工具与 400 条评测样本统计分布、G.1–G.3 算力/模型明细/真机硬件配置/3 次独立重复试验 Table 8、H 案例索引、I.1–I.5 四大环境运行时 Prompt 与约束规则），并核对项目主页全文
- **通读：** 已完成

## 原文核对笔记

- **核心动机与问题定义（§1–§2）：** 现有参数化具身策略（如 $\pi_0$、Octo、OpenVLA 等 VLA 模型及分层 VLM 系统）将感知、认知、规划与控制耦合在同一组共享参数 $\theta$ 中，导致（1）不透明耦合（Opaque coupling），难以选择性调用与归因诊断；（2）能力孤立（Capability isolation），跨系统复用困难；（3）优化纠缠（Entangled optimization），强化某一能力时梯度冲突易损害其他能力。论文提出**能力外置化（Capability Externalization）**范式：通过双层优化（Bi-level Optimization），下层在各能力专有数据集 $\mathcal{D}_m$ 上独立优化工具 $\phi_m^*$，上层优化编排策略 $\theta^*$ 在推理期按需动态调用工具集 $\mathcal{Z}=\{z_m(\cdot;\phi_m)\}_{m=1}^M$（含不调用空动作 $\perp$）。
- **具身工具协议 ETP（§3.1）：** 定义三项核心机制：（1）将工具作为标准化能力单元 $f_m(\cdot;\phi_m): X_m \to Y_m$，兼容学习模型、经典算法、可执行程序与混合模块；（2）基于能力注册表 $\mathcal{R}=\{(z_m,\rho_m,\kappa_m)\}_{m=1}^M$ 的发现与调用，其中 $\rho_m$ 描述能力与适用条件，$\kappa_m$ 约束 I/O 模式，实现策略与具体调用约定解耦；（3）隔离会话执行与运行时反馈（Isolated execution with runtime feedback），支持并发调用、远程部署及失败检测与重规划。
- **112 项具身工具库（App. D & F.1, Table 5）：** 经文献检索分解、人工筛选（轻量、易部署、环境鲁棒）、跨 2–3 个额外基准交叉验证及 API 封装，构建 112 个工具，均衡覆盖四大维度的 24 个子类别：感知与定位（36 个，如 YOLO-World、FastSAM、GigaPose、CenterPoint、Cutie、ZoeDepth、DUSt3R、Open-Fusion、Hydra、OctoMap）、认知与状态建模（25 个，如 `query_3d_scene_graph`、STM、Action Genome、Language2LTL、LINGO-Space）、推理与规划（27 个，如 Nav2、MoveIt 2、`VIRF_Main_Loop`）及执行与控制（24 个，如 AnyGrasp、Contact-GraspNet、TAPIR、R3M）。
- **EmbodiedToolBench 四维评测基准（§3.2, App. E & F.2, Table 6）：** 基于 EB-ALFRED、EB-Habitat、EB-Navigation 与 EB-Manipulation 四个环境构建共 400 条样本（每任务 100 条），评估模型的具身工具使用能力：（1）**Tool-Need Recognition**（正负样本 1:1，含关键词诱导负样本，测 Acc/F1）；（2）**Tool Selection**（基于 `text-embedding-ada-002` 余弦相似度与前置条件唯一性筛选的 4 选 1 困难负样本，测 CSR）；（3）**Tool Execution**（分 Stage 1 构造合规参数 ISR 与 Stage 2 理解工具返回并输出正确动作 AMR，联合指标为 TUSR）；（4）**Tool-Chain Composition**（覆盖 112 候选与 51 标准工具，包含约 20% 非典型依赖顺序样本，测集合 Acc/F1 与顺序一致性 OCR）。
- **主要实验与外部化边界发现（§4, Table 1–4, App. G Table 8）：** 在 4 个闭源模型（GPT-4o、GPT-5、Claude-3.7、Claude-4.6）与 4 个开源模型（Qwen3-32B、Qwen3-8B、Qwen2.5-32B、Qwen3.5-35B-A3B）上的评测表明：
  - **显著增益与开源缩小差距（RQ1）：** 工具增强在 EB-ALFRED（平均 +0.31）和 EB-Navigation（平均 +0.36，长程子任务 +0.54）上提升最大；Qwen3-8B 在 EB-ALFRED 从 0.20 升至 0.72，超过无工具 GPT-4o（0.48）；Base 子集 3 次重复实验（Table 8）确认增益稳定且方差小（闭源均值 +17.10，开源均值 +29.30）。
  - **真机实验（Table 2, App. G.2）：** 在 6-DoF 总线舵机机械臂 + 平行夹爪 + Gemini Plus RGB-D 相机 + STM32/Jetson 平台上测试桌面清理、天平操作、积木搭建三项任务（每项 10 次）。8 个模型无工具时全部为 0/10；接入工具后 GPT-4o 桌面清理达 8/10、Claude-4.6 天平操作达 9/10、Claude-3.7/4.6 积木搭建达 8/10。
  - **外置化边界（RQ3, §4.3）：** 感知（闭源 +0.26 / 开源 +0.26）与认知（+0.28 / +0.34）获益最大，而执行与精细操作（EB-Manipulation 均值仅 +0.07，Spatial 仅 +0.02；执行维度工具增益闭源仅 +0.03–+0.10）提升有限，受限于精确空间接地、接触动力学与低层控制。
  - **工具使用能力瓶颈（RQ2, Table 3 & §4.5）：** 模型普遍具备较高语法构造能力（ISR 0.94–0.96），但开源模型经常漏调工具（Qwen3-8B 召回率仅 0.31）或无法利用返回值（Qwen3-8B AMR/TUSR 仅 0.04），强模型则倾向工具链过度组合（GPT-5 / Claude-4.6 召回 0.99 但精确率降至 0.81–0.83）。错误分析表明：动作密集任务（Navigation/Manipulation）主要失误为漏调工具（36%/45%）与工具诱导偏差（Qwen3-8B 在 Manipulation 达 56.5%），感知密集任务（ALFRED/Habitat）则主要是忽略工具输出（47%/43%）。推理耗时方面，EB-ALFRED/Habitat 每任务增加约 10–30 秒，EB-Manipulation 多模型增幅超过 30 秒。

对 wiki 的映射：[Enabling Extensible Embodied Capabilities with Tools](../../wiki/entities/paper-agentic-2605-26637.md)
