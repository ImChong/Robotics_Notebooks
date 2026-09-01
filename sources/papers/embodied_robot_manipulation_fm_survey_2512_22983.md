# Embodied Robot Manipulation in the Era of Foundation Models: Planning and Learning Perspectives

- **类型**：论文（survey）
- **收录日期**：2026-09-01
- **arXiv**：<https://arxiv.org/abs/2512.22983>
- **PDF**：<https://arxiv.org/pdf/2512.22983.pdf>
- **项目页 / 列表**：<https://github.com/BaiShuanghao/Awesome-Robotics-Manipulation> — 见 [`sources/repos/awesome-robotics-manipulation.md`](../repos/awesome-robotics-manipulation.md)、[`sources/sites/awesome-robotics-manipulation.md`](../sites/awesome-robotics-manipulation.md)
- **机构（摘要口径）**：香港科技大学（广州）、中国科学院、西安交通大学、西湖大学、浙江大学、悉尼大学、北京智源人工智能研究院、北京大学等

## 一句话

从 **高层规划** 与 **低层学习式动作建模** 双轴，系统梳理基础模型时代的机器人操作：前者覆盖语言/代码/affordance/几何约束/3D 表征等结构化规划产物，后者按输入建模、潜表示学习与策略学习组织 VLA、扩散策略与多模态动作模型。

## 为什么值得保留

- **功能角色视角**：区别于按 VLA、扩散策略或生成模型族谱组织的既有综述，强调组件在 **规划链 vs 执行链** 中的接口与交互，便于跨方法对照。
- **三层分解**：高层规划 → 低层动作建模 → 执行层控制；综述主范围锁定前两层，并给出 \(A_{\text{plan}}=f_{\text{plan}}(o,l)\) 与 \(A_{\text{action}}=f_{\text{act}}(o,l\mid A_{\text{plan}})\) 的统一表述。
- **配套 Awesome 列表**：GitHub 仓库持续维护论文、代码与相关站点，适合作为操作领域 **外部索引入口**。

## 核心摘录（面向 wiki 编译）

### 与 VLA / 扩散 / 生成式综述的分工

- 既有工作多按 **模型家族** 归类（VLA survey、扩散策略 survey、生成式操作 survey）；本篇按 **规划产物 vs 可执行动作** 的功能分工重组文献。
- **机器人基础模型** 定义：在多样机器人/多模态数据上预训练、可跨任务/物体/环境迁移的模型；以 **多样预训练、广能力范围、下游可复用** 三性质刻画，而非僵化参数量阈值。

### 高层规划 taxonomy（§II）

| 类别 | 规划产物 \(A_{\text{plan}}\) 形态 | 代表 |
|------|-----------------------------------|------|
| LLM 任务规划 | 技能序列、子目标 | SayCan、Inner Monologue |
| MLLM 任务规划 | 视觉 grounded 计划 | PaLM-E、RoboBrain |
| 程序化规划 | 可执行代码 / API 调用 | Code as Policies、Statler |
| 几何约束规划 | 3D value map、关键点关系、目标位姿 | VoxPoser、ReKep、CoPa |
| Affordance 规划 | 动作相关区域/关系先验 | CLIPort、GAPartNet |
| 3D 表征规划 | 抓取候选、空间对应、优化目标 | F3RM、Splat-MOVER |

### 低层动作建模 taxonomy（§III）

- **学习策略**：RL（model-free / model-based）、IL（动作监督 / 观测监督）、辅助任务（世界建模、目标提取、多模态对齐）。
- **输入建模**：2D/3D Vision–Action、VLA（非 LLM 系 / LLM·VLM 系 / 3D VLA）、触觉/力/音频等多模态。
- **潜表示学习**：离散潜动作、连续潜动力学、隐式世界模型、Koopman 等。
- **策略学习**：MLP/Transformer、扩散/流匹配、频率域、SSM 等。

### 评测与基准（Table II 口径）

MetaWorld、RLBench、CALVIN、LIBERO、RoboCasa、SimplerEnv、RoboTwin、ManiSkill3、VLABench 等——各基准在 **长程、语言条件、真机对齐、双臂/跨本体、接触丰富度** 上各有侧重，分数须结合评测设定解读。

### 四大未来方向（§IV）

1. **通用机器人大脑**：超越分布内插值，支持未见任务结构/物体/环境的组合泛化；需灵活模态与本体接口、持续学习、规划–控制紧耦合。
2. **数据与评测瓶颈**：真机数据稀缺、仿真接触/形变保真不足、评测协议未标准化；需可复现框架、数据飞轮、高保真可微仿真。
3. **多模态物理交互**：超越视觉主导，融合触觉/音频/热等；软体/流体/颗粒等高维接触操作需新表征与物理知情学习。
4. **安全与协作**：失败预测、后门攻击、指令干扰已有初步工作；部署需自约束控制、多机协作协议、学习–经典方法（MPC/规则）混合兜底。

## 对 wiki 的映射

- 升格页面：[基础模型时代具身操作综述（Planning & Learning）](../../wiki/entities/paper-embodied-manipulation-foundation-models-survey.md)
- 交叉补强：[VLA](../../wiki/methods/vla.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[操作鲁棒性综述](../../wiki/entities/paper-robustness-robotic-manipulation-survey.md)、[VLA Survey（HMI P071）](../../wiki/entities/paper-vla-survey-embodied.md)
