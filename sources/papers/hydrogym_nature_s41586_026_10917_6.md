# The HydroGym reinforcement learning platform for fluid dynamics（Nature 2026）

> 论文来源归档（ingest）

- **标题：** The HydroGym reinforcement learning platform for fluid dynamics
- **类型：** paper / reinforcement-learning / flow-control / cfd / benchmark-platform / sim2real-transfer
- **期刊：** *Nature*（2026）
- **DOI：** <https://doi.org/10.1038/s41586-026-10917-6>
- **Nature 页：** <https://www.nature.com/articles/s41586-026-10917-6>
- **开放获取（核查结论）：**
  - **arXiv 预印本（推荐全文入口）：** <https://arxiv.org/abs/2512.17534> · PDF：<https://arxiv.org/pdf/2512.17534.pdf> — **完全开放**
  - **Nature 读者共享链（作者宣传 #OpenAccess）：** <https://rdcu.be/fBvqa> — 免订阅阅读 Nature 排版版（链路易变，以 Nature 页为准）
  - **补充材料（SI PDF）：** <https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10917-6/MediaObjects/41586_2026_10917_MOESM1_ESM.pdf>
  - **前序会议版（L4DC 2025 / PMLR）：** HydroGym: A Reinforcement Learning Platform for Fluid Dynamics — 与 Nature 版为同一平台主线，arXiv bib 中 `@inproceedings{lagemann2025hydrogym_a}` 指向该会
- **项目页：** <https://dynamicslab.github.io/hydrogym/>
- **代码：** <https://github.com/dynamicslab/hydrogym>（MIT）
- **环境 checkpoint：** <https://huggingface.co/datasets/dynamicslab/HydroGym-environments>
- **机构：** 华盛顿大学（UW）/ AI Institute in Dynamic Systems；密歇根大学（UMich）；慕尼黑工业大学（TUM）；RWTH Aachen；KTH；仁荷大学（Inha）等
- **通讯作者：** Christian Lagemann、Ricardo Vinuesa、Steven L. Brunton
- **入库日期：** 2026-09-02
- **一句话说明：** 提出 **solver-independent** 的 **Gymnasium 兼容** 主动流控 RL 平台 **HydroGym**：**61+** 验证环境、**6** 类 CFD 后端（FEM / LBM / FV / 谱元 / 可微 JAX），系统展示 **PPO/DDPG/TD3** 基线与 **GPPO/MARL/迁移学习**；核心亮点为 **通道湍流代理训练 → 三维翼型零样本减阻 38%**（\(Re_c=200{,}000\)），探索代价较直接翼型训练降约 **4 个数量级**。

## 核心摘录（面向 wiki 编译）

### 1) 问题与平台定位

- **要点：** 流体控制高维、非线性、多尺度；RL 在蛋白折叠、聚变等离子体等领域靠 **共享 benchmark** 突破，而流控长期陷「单几何单工况调参」的 **specificity trap**，难以累积、迁移与公平对比。HydroGym 提供统一 **Gymnasium** 接口、**61+** 验证环境与 **6** 种求解器后端，Re 上至 **\(4\times10^5\)**，含 2D/3D 与可压缩工况。
- **对 wiki 的映射：** [`wiki/entities/paper-hydrogym.md`](../../wiki/entities/paper-hydrogym.md)

### 2) 求解器与后端分工

- **要点：** **Firedrake（FEM）** 快速原型；**MAIA LBM / MAIA FV** GPU 大规模湍流；**NEK5000** 谱元高 Re 不可压通道；**JAX / JAX-Fluids** 可微求解支撑 **GPPO**（梯度增强 PPO，Kolmogorov 流训练迭代数降 **≥65%**）；**MPI** 并行与 **PettingZoo** 多智能体接口（如 3D 圆柱 ZNMF 射流 spanwise 分布控制）。
- **对 wiki 的映射：** 同上实体页「核心结构 / 工程实践」

### 3) 基准演示与物理机制

- **要点：** 四类代表环境：**Fluidic pinball**（\(Re=100/150\)，协调旋转，~90% 减阻）、**Open cavity**（剪切层/声学反馈抑制）、**Cylinder**（\(Re=3900\)，ZNMF 边界层操纵）、**Gust–airfoil**（NACA0012，\(\alpha=20°\)，载荷振荡降 ~20%）。基线训练累计 **>150,000 GPU·h**（SI §4）。
- **对 wiki 的映射：** 同上实体页「实验要点」

### 4) 迁移与零样本翼型部署

- **要点：** Re 缩放、圆→方柱几何、2D→3D **微调** 约减半 episode；**零样本**：仅在 **\(Re_\tau=206\)** 湍流通道上训练 **MARL（TD3）**，直接部署到 **\(Re_c=200{,}000\)** 三维 NACA0012，局部皮肤摩擦降 **~38%**、总阻力降 **~11%**，优于 opposition control / uniform blowing；相对直接翼型训练，探索效率 **>10⁴**（网格与并行度差异）。
- **对 wiki 的映射：** 同上实体页；交叉 [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)、[`wiki/entities/gymnasium.md`](../../wiki/entities/gymnasium.md)

### 5) 开源边界（项目页 / GitHub 核查，截至 2026-09-02）

- **已开源：** `dynamicslab/hydrogym`（MIT）；文档站 `dynamicslab.github.io/hydrogym`；Docker 镜像（CUDA/ROCm）；HuggingFace 环境网格/checkpoint **按需下载**；示例含 Firedrake / MAIA / NEK / JAX 各后端 **getting_started** 与 SB3 训练脚本。
- **边界：** 大规模 MAIA/NEK 后端依赖 GPU/HPC 与专有栈；离线 HPC 节点需 **预下载** HF checkpoint；**非** 机器人本体仿真，但与 RL **环境 API / 基准文化** 同构。
- **对 wiki 的映射：** [`sources/repos/dynamicslab_hydrogym.md`](../repos/dynamicslab_hydrogym.md)、[`sources/sites/dynamicslab_hydrogym.md`](../sites/dynamicslab_hydrogym.md)

## 当前提炼状态

- [x] Nature / arXiv / 项目页 / GitHub 开放获取与源码核查
- [x] wiki 映射：`wiki/entities/paper-hydrogym.md` 新建
