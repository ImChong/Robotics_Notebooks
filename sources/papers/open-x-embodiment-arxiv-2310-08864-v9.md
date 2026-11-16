# Open X-Embodiment: Robotic Learning Datasets and RT-X Models — arXiv 2310.08864v9 一手阅读

- **类型：** paper / primary source
- **英文标题：** Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- **中文标题：** Open X-Embodiment：跨本体机器人学习数据集与 RT-X 模型
- **作者：** Open X-Embodiment Collaboration（完整作者名单以 arXiv 为准）
- **初稿：** 2023-10-13
- **阅读版本：** v9，2025-05-14
- **原文：** https://arxiv.org/abs/2310.08864v9
- **全文 HTML：** https://arxiv.org/html/2310.08864v9
- **PDF：** https://arxiv.org/pdf/2310.08864v9
- **项目页：** https://robotics-transformer-x.github.io/
- **官方仓库：** https://github.com/google-deepmind/open_x_embodiment
- **核对日期：** 2026-11-08
- **核对范围：** 阅读 v9 全文主体的 Abstract、I Introduction、II Related Work、III-A/B 数据、IV-A/B/C 模型/训练推理、V-A/B/C 实验与消融、VI Discussion/Future Work，查核 Fig. 1–4、Table I–II 和参考文献入口；对照项目主页与官方仓库 README。参考文献条目不等同于逐篇通读其引用论文。
- **通读：** 已完成

## 论文问题与贡献

目标是验证跨机器人、跨机构混合模仿数据是否可以产生正迁移，并为后续跨本体研究提供共享数据基础设施。论文强调创新重点是 **数据规模/异构混合与迁移证据**，而非另创策略网络。

- **公开数据总库：** 60 个既有数据集、22 种机器人本体、100 万+ 真机轨迹；正文称来自 34 个机器人实验室，摘要称由 21 家机构合作汇集；两种统计口径不可互换。
- **技能：** 527 种技能，160,266 个任务；对象与技能主要覆盖操作，包括抓放，也存在清洁与装配等长尾。
- **模型实验所用数据：** 当时从 9 种机械臂选出的训练混合，而非全部 22 种本体。论文说明数据总库持续扩展，二者快照不同。
- **模型：** RT-1-X（仅机器人混合数据）；RT-2-X（机器人混合数据与预训练 VLM 数据约 1:1 的 co-fine-tuning），均以图像、任务文本预测 token 化的动作。

## III：数据与粗对齐接口

源数据通过 **RLDS episode/step** 存储（序列化 TFRecord）。不同本体可有多个 RGB、深度图或点云；RT-X 实验只选择每数据集的一个典型相机视角，统一图像分辨率，并转为 7 维末端动作：`x,y,z,roll,pitch,yaw,gripper`（或这些量的变化率）；各数据集先归一化再离散化。

**重要限制：** 相机外参、动作坐标系并没有跨数据集完全对齐；绝对位置、相对增量、速度可以继续沿用原始控制语义，同一数值在两个机器人上不保证代表同样物理运动。RLDS 统一存储格式不等于全本体统一动力学。

## IV：RT-X 架构、训练和执行

1. **RT-1-X：** 约 35M 参数；EfficientNet + FiLM 处理图文输入、TokenLearner/Transformer 表示；历史 15 帧；decoder-only Transformer 预测离散动作。
2. **RT-2-X：** 采用 RT-2 PaLI-X/VLM 方向，将动作离散码视为可输出文本 token，借视觉语言预训练作控制迁移。
3. **监督与输出：** 两模型使用分类交叉熵；动作按 8 个维度离散（7 维末端量 + 1 维 episode termination，每维 256 bins）。这与仓库示例用于内部预测的 11 action tokens 并非同一口径，不应混用。
4. **运行：** 论文中的 RT-1 本地推理，RT-2 云端推理，经网络查询；按硬件需要约 3–10 Hz。官方仓库示例本身只演示离线推理，不是云端/真实机器人的闭环部署。

## V：实测与消融（切勿混淆指标）

论文统计 **3,600 次评测、6 种机器人**。

| 评测 | 模型/对照 | 结果 | 解释边界 |
|---|---|---|---|
| 五个小数据域 | RT-1-X 对原方法 | 其中 4/5 领先；论文汇总均值约 50% 相对提升 | 不代表五域每一域都提升 50% |
| 大数据域 Bridge / RT-1 | RT-1-X | 27% / 73% | 对应单本体 RT-1 基线在相关实验是 40% 或 30% / 92%；高容量仍重要 |
| 新技能，Google Robot | RT-2-X 55B vs RT-2 55B | **75.8% vs 27.3%** | 约 2.78 倍，属 Bridge 数据中有、Google Robot 训练数据中没有的技能 |
| 去掉 Bridge 数据后的新技能 | RT-2-X 55B | **42.8%** | 表明跨 WidowX 数据贡献的重要性 |
| RT-2 泛化综合评测 | RT-2-X 55B vs RT-2 55B | **61% vs 62%** | 未见物体/背景/环境的综合指标，非物体与背景各一项 |
| RT-2-X 5B 历史消融（新技能） | 2 帧 vs 0 帧 | **44.4% vs 14.5%** | 其他配置按论文 Table II |
| RT-2-X 5B 预训练消融（新技能） | Web-pretrained vs 从零训练 | **44.4% vs 0%** | 不能把此消融推到所有模型 |

Table I 大数据域精细对照：Bridge 在 IRIS / RAIL 两处 RT-1-X 都为 27%；RT-1 为 40% / 30%；RT-2-X 55B 为 50% / 30%。Google Robot 6 skills: RT-1 92%、RT-1-X 73%、RT-2-X 91%。各行不是相同容量模型的无条件公平对比。

## VI：局限与后续方向

- 研究聚焦机器人操作，未覆盖截然不同的传感/执行模态；**没有测试迁移到训练数据外全新机器人**。
- 未建立“什么时候正迁移必然发生”的判据；大数据域 RT-1-X 发生欠拟合。
- 作者提供数据、使用示例与 RT-1-X checkpoint 作为研究底座，但不等于公开 RT-2-X 端到端训练与部署系统。

## 对 wiki 的映射

- 主实体：[Open X-Embodiment / RT-X](../../wiki/entities/paper-open-x-embodiment.md)：数据/实验口径、Mermaid 模型流程图与源码时序图。
- 概念：[Open X-Embodiment](../../wiki/concepts/open-x-embodiment.md)：统一数据与动力学仍异构的区别。
- 相关官方资料：[项目页归档](../sites/open-x-embodiment.md)、[仓库源码归档](../repos/open-x-embodiment.md)。
