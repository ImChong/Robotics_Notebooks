# SLA: Scaling Spatial-Language-Action Model via Continual Pre-training / 通过持续预训练扩展空间-语言-动作模型

- **类型：** 原始论文（arXiv v1，2026-11-05）
- **作者：** Fuhao Li 等
- **论文：** https://arxiv.org/abs/2611.08373
- **全文（HTML v1）：** https://arxiv.org/html/2611.08373v1
- **论文 PDF：** https://arxiv.org/pdf/2611.08373
- **官方代码：** https://github.com/OpenHelix-Team/Spatial-Forcing/tree/master/SLA
- **官方 README：** https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/README.md
- **官方项目页：** https://sla-team.github.io/omnisf_projectpage/ （已读取）
- **作者备用项目入口：** https://embodiedsla.github.io/ （已读取，同一项目）
- **Hugging Face 论文页：** https://huggingface.co/papers/2611.08373
- **作者模型/数据合集：** https://huggingface.co/collections/haofuly/sla-collection
- **资料复核：** 2026-11-08，新增作者双项目页、HF 论文索引与 4 个官方资源条目的一手交叉验证
- **入库日期：** 2026-11-07
- **核对范围：** 通读 arXiv 官方 HTML v1 正文中的研究动机、方法、数据与预训练、各评测基准、消融及局限，并检查对应附录；逐项对照官方 GitHub SLA/README 与关键训练、推理源码。使用官方 HTML 全文作为阅读底本，未将摘要或二手报道充作全文。
- **通读：** 已完成

## 研究问题

已有 VLA 能结合图像、语言与动作，却容易依赖外观关联与本体状态，跨场景空间泛化仍有瓶颈。SLA 关注在沿用 π0.5 动作策略的前提下，能否通过大规模异构机器人数据的持续预训练注入 3D 几何归纳偏置，而不增加部署期的空间教师模型开销。

## 方法与训练证据

1. 以 π0.5 为可训练策略主干，统一来自不同机器人的末端位姿与动作表示（统一坐标和 SO(3) 相对旋转等），并在预训练中使用 State Dropout。
2. 使用冻结的 OmniVGGT 从多视角视觉输入提取空间特征作为教师监督。训练期从 VLA 中间层提取视觉 token，以投影/融合模块向教师特征对齐；论文提出基于输入的层选择/门控，联合动作与空间对齐目标。
3. 训练优化仍包含 flow-matching 动作监督，概念上可写为 L_total = L_action + α × L_spatial；空间教师与对齐模块不进入常规部署的动作生成路径。
4. 持续预训练后使用标准下游示范进行后训练；论文对预训练与后训练是否都做空间对齐展开消融。

## 数据集及规模口径

论文列示六来源（具体单位以论文原表为准）：InternData-A1、AgiBotWorld-Beta、RoboCasa365、RoboCOIN、DROID 与 MolmoAct。原始汇总条目约 171.39 万段、14,330 小时，为**来源数据统计**，不能直接等同实际清洗/抽样后的训练时长。

## 评测（作者报告，非独立复现）

| 基准 | SLA 指标 | 比较与说明 |
|---|---|---|
| LIBERO-Plus | 87.4% | π0.5 84.4%；Spatial Forcing 85.1%，须对齐各论文设置 |
| RoboTwin 2.0 Clean2Random | 72.3% | π0.5 46.0%；Spatial Forcing 62.6% |
| RoboTwin 2.0 Clean2Clean | 86.0% | 对应平均值 79.2%（两设置平均） |
| RoboCasa-Panda | 68.4% | Spatial Forcing 67.6%；数据量/任务配置需核对 |
| RoboDojo-Real | 33.9% | π0.5 12.8%；该评测采用历史视觉帧的记忆机制 |
| 8 组真实部署条件 | 平均 83.8% | π0.5 59.8%，提升 24.0 个百分点；作者报告 |

关键消融：RoboTwin Clean2Random 上两阶段空间对齐 72.3%，仅预训练 63.2%，仅后训练 57.1%，均不对齐 48.2%；论文的自适应层融合与固定层比较也有提升。上述数值不是在本次 ingest 环境重新训练测得。

## 开源与论文实现的边界

- 官方 SLA/README 公布 LIBERO-Plus、RoboTwin 的后训练与评估、训练入口、预训练权重、训练组件及若干基准检查点。
- README 同时标记 RoboDojo-Real 后训练/评测与持续预训练数据仍待开放；不能把代码可见等同所有原始数据可复现。
- 当前公开的示例配置 train_libero_plus.py 与 pretrain.py 中，vla_adapt_layers_align=False、vla_layers_align=12；**默认代码配置是固定层选择**，不能直接宣称其默认已启用论文所述自适应门控。复现论文门控消融需进一步核对配置与实验细节。
- RoboDojo-Real 的历史帧记忆机制是基准特定推理设定；“部署期没有额外空间教师”不代表所有评测配置具有完全相同的输入/计算成本。

## 2026-11-08 项目主页、训练规模和复现资源复核

- 已打开论文 HTML v1 的方法（§3）、数据（§3.3）、实验（§4）、消融（§4.4）和附录（A、B），并对照作者两处项目主页；二者目前均能展示方法介绍、数据表与 4 个主基准成绩。原「站点无法核验」的表述不再适用。
- **实际训练开销（论文 §4.2）：** 持续预训练 36k 步、effective global batch 4,608，48× H100 约 110 小时；不等于「14,330 小时」的**原始数据源统计**，后者见论文 Table 1。
- **正式协议：** LIBERO-Plus 10,030 rollouts、七扰动维度按样本数加权；RoboTwin 2.0 共 50 任务，每任务 50 条 clean 示范，Clean2Clean 与 Clean2Random 各 100 trials/task；RoboCasa-Panda 24 任务、每任务 50 demonstrations、每任务每种子 50 trials×3 seeds。
- **论文可控消融：** §4.4 自适应层选择 72.3% Clean2Random vs 固定 12 层 67.3%，状态 Dropout 25% 为 72.3% vs 0% 为 20.7%，均为该 RoboTwin 设置下的作者结果；公开示例配置默认 `vla_adapt_layers_align=False` 不能等同论文自适应版本。
- **HF 资源与完整语料不等同：** 作者 Collection 目前含 `SLA_base`、`SLA_training_component`、`SLA_benchmark_ckpts` 和 `SLA_real_world_demo` 四项；后者是演示数据。官方 README 同时标记「continual pre-training data」未发布。
- **边界：** HF 论文页为同一 arXiv 论文的社区索引，不是重复工作；未下载权重或运行代码，评测均沿用作者论文与官网原始报告。

## 对 wiki 的映射

- 实体页：[SLA / 通过持续预训练扩展空间-语言-动作模型](../../wiki/entities/paper-sla-spatial-language-action.md)
- 技术知识链：[VLA 与基础策略](../../wiki/overview/hub-vla.md)
- 前作：[Spatial Forcing](../../wiki/entities/paper-rcl-ref-0f2536c81a1992e3c3b8-spatial-forcing-implicit-spatial-representation.md)
