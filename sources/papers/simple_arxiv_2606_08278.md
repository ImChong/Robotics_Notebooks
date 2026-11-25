# simple_arxiv_2606_08278

> 来源归档（ingest）

- **标题：** SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation
- **类型：** paper
- **来源：** arXiv:2606.08278v2（2026-10-12；v1：2026-06-06）；项目页 <https://psi-lab.ai/SIMPLE>
- **机构：** 南加州大学 Physical Superintelligence (PSI) Lab
- **作者：** Songlin Wei, Zhenhao Ni, Jie Liu, Zhenyu Zhao, Junjie Ye, Hongyi Jing, Junkai Xia, Xiawei Liu, Michael Leong, Liang Heng, Di Huang, Yue Wang（* equal contribution, † corresponding author）
- **代码：** <https://github.com/physical-superintelligence-lab/SIMPLE>（已开源）
- **数据集：** Hugging Face `USC-PSI-Lab/psi-data`（含 `simple-eval` 评测环境）
- **入库日期：** 2026-07-16
- **复核日期：** 2026-11-21（v2 全文 + 官方项目页 + GitHub 主分支源码）
- **核对范围：** arXiv:2606.08278v2 HTML 全文 Abstract、§1–5（架构、数据管线、基线评测、迁移、Limitations）、附录 A.1–A.5（类与运行时、资产、实验配置、AMO/SONIC、16 任务多任务评测）；GitHub README 与评测 CLI 入口。
- **通读：** 已完成
- **一句话说明：** USC PSI Lab 提出的 **人形全身 loco-manipulation 统一仿真 testbed**：MuJoCo 接触物理 + Isaac Sim 光追渲染双引擎，论文 v2 基准 16 任务（官网与 v2 §3.2 仍称 60）/ 50 室内场景 / 1000+ 物体，内置运动规划与 VR 遥操作数据采集，并大规模 benchmark VLA/WAM/IL 策略；仿真排序与真机强相关，支持纯仿真数据 **零样本 sim-to-real**。

## 论文 v2 版本核对（2026-11-21）

- **一手资料：** [arXiv:2606.08278v2 全文](https://arxiv.org/html/2606.08278v2)（2026-10-12 修订）与 [官方项目页](https://psi-lab.ai/SIMPLE/)。下方早期 v1/官网摘录保留溯源价值，遇到不一致以本节注明的版本为准。
- **任务计数矛盾须标注：** v2 Abstract、Introduction、Conclusion 与 Appendix A.5 报 **16 项评测任务**；v2 §3.2 却仍写 **60**，官网亦显示 **60**。附录 Table 7 枚举 16 项代表任务；不能直接写成“v2 测完 60 项”。
- **评测组不同：** v2 Table 1 是 **7 个模型（Ψ₀、π₀.₅、GR00T N1.6、DreamZero、Cosmos3、Diffusion Policy、ACT）× 6 个单任务**，各 level 每项 **10 次**；Appendix Table 8 是 **5 个 foundation models × 16 项多任务**，通常各 level **20 次**，DreamZero **10 次**；官网还展示旧版 **9 模型**（含 InternVLA/H-RDT/EgoVLA）的结果，不能合并排名。
- **政策 I/O：** v2 §3.5 定义 **32D 本体状态**（手 14 + 臂 14 + 腰 3 + 底盘高度 1）及 **36D 动作**（32D 上身关节目标 + 底盘 4D `vx, vy, vyaw, qyaw`）。VR 采集为 **50Hz** 并转为 LeRobot 格式。
- **域随机化：** 正式表格以 **Level 0/1/2** 统计；Level 0 视觉/干扰物，Level 1 进一步改变材质与光照，Level 2 进一步变化物体/机器人空间位姿。v2 §3.5 行文用 1/2/3 编号，与表格存在标号不一致。
- **新增 sim-to-real 实例：** v2 Table 5 的 XMoveBendCarryBox **7/10 仿真 → 5/10 真机**；另有 Pick & Place **9/10→8/10**、Handover **10/10→8/10**，均为相似场景下、无真机 fine-tune 的有限示例，不表示开放域普适迁移。
- **多任务结果：** v2 Table 8 的 L0/L1/L2 平均成功率（%）：Ψ₀ **64.4/59.7/48.8**；DreamZero **64.4/58.1/46.9**。精细接触任务（OpenFaucet）成功率仍低。
- **局限：** 论文 §4.6 与 §5 明确同步 Isaac Sim 光追约 **4fps/GPU**（可在离线生成中通过并行缓解），只对刚体建模，布料、绳索等形变对象不适用。
- **代码与数据新发布：** 官方 README 2026-11-11：main 是 **Isaac Sim 4.5 / MuJoCo 3.3**，另有 `isaacsim-6.2-mujoco-3.12` 分支；发布 **12 Teleop + 5 MP** decoupled-WBC 数据集及 Ψ₀ checkpoints。17 是公开数据集任务数，与论文 16 任务多任务评测分开统计。

## 核心论文摘录（v1/官网历史摘录，以 v2 核对节为准）

### 1) 问题设定与系统定位（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2606.08278>
- **核心贡献：** 人形 foundation model 进展快于可复现评测；真机评测贵、难 reset、难公平横比；现有仿真 benchmark 多聚焦 **桌面/轮式**，缺 **全身行走+操作+平衡** 一体化 testbed。SIMPLE 提供 **full-stack** 仿真基础设施：评测、数据采集、策略学习接口统一。
- **对 wiki 的映射：**
  - [SIMPLE 论文实体](../../wiki/entities/paper-loco-manip-161-075-simple.md)
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
  - [Loco-Manipulation 任务页](../../wiki/tasks/loco-manipulation.md)

### 2) 双仿真器架构与全身控制解耦（Sec.3.1）

- **链接：** <https://arxiv.org/html/2606.08278v1#S3.SS1>
- **核心贡献：**
  - **MuJoCo** 负责刚体动力学、接触解析与高频控制；每步将状态同步至 **Isaac Sim** 做光追渲染——物理与视觉严格解耦。
  - **高层策略**（VLA/WAM/IL）预测上身运动学与底座导航指令；**底层 tracking controller**（AMO/SONIC 等）高频维持平衡与 locomotion。
  - 整体封装为 **OpenAI Gym** 接口（如 `G1WholebodyXmovePickTeleop-v0`），原生兼容 Ψ₀、π₀.₅、GR00T-N1.6、DreamZero 等主流架构。
- **对 wiki 的映射：**
  - [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)
  - [VLA](../../wiki/methods/vla.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)

### 3) 规模化资产与两条数据采集管线（Sec.3.2–3.4）

- **链接：** <https://arxiv.org/html/2606.08278v1#S3>
- **核心贡献：**
  - **60** 全身任务（刚性取放、非抓取交互、铰接物体操作等），**50** 室内场景，**1000+** 物体资产；累计 **6000+** 轨迹。
  - **运动规划管线：** 物体落稳姿态 → **BoDex** 合成抓取 → **CuRobo** 双臂运动学轨迹 + 脚本化底座协调。
  - **VR 遥操作：** PICO XR 低延迟 egocentric 双目流；手部 IK retarget，全身平衡/locomotion 由 tracking policy 自主管理。
  - **采集效率（项目页 Table 2）：** 全身 pick-place 任务上，仿真内遥操作 **310.3 demos/hr**（最快），真机遥操作 **206.8**，运动规划 **58.9**。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Psi0（161 策展条目）](../../wiki/entities/paper-loco-manip-161-156-psi0.md)

### 4) 三级域随机化评测与大规模 policy benchmark（Sec.4）

- **链接：** <https://psi-lab.ai/SIMPLE/>（Benchmark 表）
- **核心贡献：**
  - **L0 / L1 / L2** 渐进 OOD 域随机化；每任务 **10** 次 rollout 报成功率。
  - **9** 类代表策略：**Ψ₀、GR00T N1.6、π₀.₅、InternVLA、H-RDT、DreamZero、EgoVLA、Diffusion Policy、ACT**；跨 **6** 任务族（XMovePick、BendPick、Handover、Mobile P&P、Grasp、XMoveBendPick）。
  - 项目页示例：Ψ₀ 在多数任务 L0/L1 接近满分；GR00T N1.6 在 Mobile P&P 等任务 L2 骤降；**仿真排序与真机实验高度一致**。
- **对 wiki 的映射：**
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)

### 5) 消融与零样本 sim-to-real（Sec.5 / Transfer）

- **链接：** <https://psi-lab.ai/SIMPLE/>（Analysis & Transfer）
- **核心贡献：**
  - **域随机化混合训练**（5×L0 + 5×L1）相对纯 L0 在更难评测集上泛化更好；**遥操作数据 scaling**（10→100 轨迹）显著提升 XmoveBendPick 成功率（0.50→1.00 @ Set 0）。
  - **数据来源：** 纯运动规划数据 avg **5.00/10**；纯遥操作 **7.56/10**——人类演示质量对 VLA 微调更关键。
  - **零样本 sim-to-real：** 仅在 SIMPLE 数据上微调的单一策略，Pick & Place **Sim 0.90 / Real 0.80**，Handover **Sim 1.00 / Real 0.80**，**无真机 fine-tune**。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)

## 其他公开资料（非 PDF 正文）

- **项目页（交互 benchmark 表、管线图、sim-to-real 并排视频）：** <https://psi-lab.ai/SIMPLE/> — 归档见 [sources/sites/psi-lab-simple.md](../sites/psi-lab-simple.md)
- **代码仓库：** <https://github.com/physical-superintelligence-lab/SIMPLE> — 归档见 [sources/repos/simple_usc_psi.md](../repos/simple_usc_psi.md)
- **161 篇策展初稿（描述有误，以本文为准）：** [loco_manip_161_survey_075_simple.md](loco_manip_161_survey_075_simple.md)

## 当前提炼状态

- [x] 论文摘要与核心方法摘录（≥5 条）
- [x] 项目页源码开放核查（GitHub + HF eval 数据）
- [x] wiki 页面映射与交叉链接
