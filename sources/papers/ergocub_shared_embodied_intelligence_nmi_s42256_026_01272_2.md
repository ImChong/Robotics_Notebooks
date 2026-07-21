# Towards shared embodied intelligence in humanoid robots …（ergoCub，*Nat Mach Intell* 2026）

> 来源归档（ingest）

- **标题：** Towards shared embodied intelligence in humanoid robots through optimization, development and testing of the human-aware ergoCub robot
- **缩写：** **ergoCub** / **Shared Embodied Intelligence**
- **类型：** paper / humanoid / hardware-codesign / physical-hri / whole-body-control / ergonomics
- **期刊：** *Nature Machine Intelligence*（2026）
- **DOI：** <https://doi.org/10.1038/s42256-026-01272-2>
- **Nature 页：** <https://www.nature.com/articles/s42256-026-01272-2>
- **项目页：** <https://ergocub.eu/>
- **AMI 专题页：** <https://ami.iit.it/en/human-robot-collaboration>
- **论文代码仓：** <https://github.com/ami-iit/paper_sartore_2025_ergocub_nature_machine_intelligence>（BSD-3-Clause；Zenodo <https://doi.org/10.5281/zenodo.17011716>）
- **依赖库：** [adam](https://github.com/ami-iit/adam)（可微刚体动力学）；[shared-controllers](https://github.com/ami-iit/shared-controllers) → 现指向 [gbionics/shared-controllers](https://github.com/gbionics/shared-controllers)
- **机构：** 意大利技术研究院（IIT）Artificial and Mechanical Intelligence / Humanoid Sensing and Perception / iCub Tech；GenerativeBionics；意大利国家工伤保险研究所（INAIL）；曼彻斯特大学
- **通讯作者：** Carlotta Sartore、Daniele Pucci
- **入库日期：** 2026-07-21
- **一句话说明：** 提出 **shared embodied intelligence** 架构：在人–机耦合动力学上联合优化人形 **硬件连杆长度** 与 **物理智能（分层控制）参数**，并以人体工程学指标（尤其 L5–S1 背应力）与行走性能为目标，落地 **ergoCub** 真机协作搬举与行走。

## 摘录 1：问题与贡献

- **动机：** 协作依赖 **shared intelligence**（对伙伴的内部表征与协调）；智能又受 **embodied cognition**（形态与物理智能共进化）约束。现有人形 cascade WBC / codesign 往往把硬件当给定、或只优化子部件，且鲜少把 **人体模型** 嵌进硬件与控制联合优化。
- **科学贡献：** 可微参数化浮动基人–机模型（连杆长度等硬件参数保持惯性一致）+ 参数化机器人物理智能（增益、规划时域等），并嵌入人类运动策略表征。
- **技术贡献：** 以 iCub3 为起点制造的 **ergoCub**（ergo + Cub），在 **行走 + 协作搬举** 上相对人体工程学与步态指标优化。
- **注意：** 优化目标是 **生物力学风险指标**（如背力矩），**不一定**对齐主观舒适/疲劳感。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-ergocub-shared-embodied-intelligence.md`](../../wiki/entities/paper-ergocub-shared-embodied-intelligence.md)；交叉 [Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Humanoid Robot](../../wiki/entities/humanoid-robot.md)、[iCub3 Avatar 计划页](../../wiki/entities/paper-notebook-icub3-avatar-system-enabling-remote-fully-immers.md)。

## 摘录 2：双实例架构（硬件 → 物理智能）

- **实例 A（最优硬件）：** 用球体/圆柱/方体简化 iCub3 连杆，参数化质量/惯性；构建人–机–载荷 **耦合动力学**；在多组 **静态搬举构型** 上最小化双方关节力矩（强调人体背应力），并抬高 CoM 以增大系统带宽、改善行走鲁棒性；高度上界保证可制造性。
- **实例 B（物理智能）：** 以制造后的 ergoCub 为固定硬件，优化控制侧增益/权重等；人与机器人均用对称的分层 **trajectory generation → adjustment → control**；协作时在线估计人体运动学/动力学并更新内部人体模型。
- **选型结果：** 比较「蓝」（载荷高度 0.8–1.2 m）与「绿」（0.8–1.5 m）优化解后采用绿色解制造；肢体长度用金属延长件逼近优化值。

**对 wiki 的映射：** 实体页「流程总览 / 方法栈」；与经典「先定硬件再调控制」对照。

## 摘录 3：真机结果要点

- **协作搬举（空 / 1 kg / 2 kg）：** 机器人跟人手高度（平均误差约 0.0084 m）；L5–S1 峰值力矩相对独自搬举显著下降（文中例：空载约 43.95→24.88 Nm；1 kg 约 50.66→25.77 Nm；2 kg 约 44.88→31.82 Nm）。
- **跟随时延：** 控制未显式预测人体运动，存在滞后；头部 LCD 按 L5–S1 应力高低显示表情反馈。
- **行走 vs iCub3（同命令与控制架构）：** 最大步长约 **0.35 m vs 0.28 m**；最小步周期约 **0.5 s vs 0.8 s**；可在持续/时变扰动与额定载荷（文中 6 kg）下做足迹调整。
- **接受度：** 850 名工业/医疗领域受试者问卷中，ergoCub 评价高于 Baxter 与 R1（Supplementary）。

**对 wiki 的映射：** 实体页「实验要点」；[Locomotion](../../wiki/tasks/locomotion.md) 交叉引用。

## 摘录 4：局限与 Sim2Real

- 非线性优化对初值敏感 → 以 iCub3 初始化。
- 硬件阶段用静态构型；动态人因主要交给物理智能层。
- 制造与优化模型存在差距：LCD 头增大高度；密度均匀假设 vs 复用 iCub3 电子/电机 + 延长件 → 实机约 **56.70 kg**（优化简化模型曾估 ~70 kg）；腿/躯干相对优化解约短 0.01 m。
- 交互当前偏 **反应式**；讨论建议未来接入短时域人体运动预测，而不改硬件优化阶段。
- 未做长期主观舒适/临床人体工程学大规模验证。

**对 wiki 的映射：** 实体页「局限与风险」；[Sim2Real](../../wiki/concepts/sim2real.md) 作制造–模型缺口交叉。

## 摘录 5：开源边界（项目页 / GitHub 核查，截至 2026-07-21）

- **项目页** <https://ergocub.eu/>：介绍整机、可穿戴 iFeel、场景与新闻；**未**在页首直接列出论文复现代码 GitHub（需以论文/AMI/代码仓为准）。
- **已开源：**
  - 论文复现仓 `ami-iit/paper_sartore_2025_ergocub_nature_machine_intelligence`：`optimal_hardware/`、`physical_intelligence/`、`data/` + Docker recipe（硬件优化）；BSD-3-Clause。
  - `ami-iit/adam`：JAX/CasADi/PyTorch/NumPy 可微浮动基动力学（`pip install adam-robotics[...]`）；BSD-3-Clause。
  - `ami-iit/shared-controllers` → `gbionics/shared-controllers`：`wholebodycontrollib` + 真机/仿真脚本（YARP / iDynTree 等）。
- **边界：** 真机闭环依赖 YARP/iCub 生态与可穿戴传感；非整机 CAD/BOM 随论文仓发布。结论：**已开源（论文脚本 + 核心库）**；整机硬件图纸不在本次公开复现范围。

**对 wiki 的映射：** sites/repos 归档与实体页「开源状态 / 源码运行时序图」。

## 当前提炼状态

- [x] Nature 正文 / 项目页 / 论文仓 / adam / shared-controllers README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-ergocub-shared-embodied-intelligence.md` 新建
- [x] 开源边界写入 sites / repos / wiki 局限
