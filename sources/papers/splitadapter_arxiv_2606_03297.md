# splitadapter_arxiv_2606_03297

> 来源归档（ingest）

- **标题：** SplitAdapter: Load-Aware Humanoid Loco-Manipulation via Factorized Adaptation
- **类型：** paper
- **来源：** arXiv:2606.03297（preprint，2026）
- **机构：** Future Robot AI Group, Samsung Electronics
- **入库日期：** 2026-06-04
- **一句话说明：** 在 **冻结 PhysHSI 类 AMP 搬箱策略** 之上，用 **物体/负载** 与 **动力学** 双分支历史编码器 + **分裂世界模型** + **GRL 交叉对抗解耦** + **分层 FiLM**，在 2/4/6 kg 与 0/30/60 cm 搬放高度下提升 MuJoCo sim-to-sim 与 **Unitree G1 零样本真机** 全流程成功率，重载（6 kg）增益最大。

## 核心论文摘录（MVP）

### 1) 问题设定与动机（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2606.03297>
- **核心贡献：** 人形 loco-manipulation 在 **载荷变化**（质量、搬放高度）与 **机器人/环境动力学失配** 同时作用时，sim-to-real 尤其难；现有 **单 latent 历史适配器**（外参估计、特权重建、统一世界模型 FiLM）易把 **负载相关信号** 与 **残差动力学** 混在同一表示里，重载搬运时鲁棒性下降。SplitAdapter **不重训全身控制器**，而是在预训练策略上学习 **因子化在线适配模块**。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [SplitAdapter 论文实体](../../wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)

### 2) 冻结基策略与双分支历史编码（Sec. 3.1–3.2）

- **链接：** <https://arxiv.org/html/2606.03297>
- **核心贡献：** 基策略为 **PhysHSI 风格 AMP RL 搬箱策略**（目标关节角 + 低层 PD）；训练后 **冻结**，仅优化适配器。共享历史编码 \(f_{\mathrm{hist}}\) 从最近 \(H\) 步 \((o,a)\) 得 \(e_t\)，再分叉为：**物体/负载头** \(f_{\mathrm{obj}}\) → \(z_{\mathrm{obj},t}\)、质量估计 \(\hat m_t\)、装载状态 \(\hat \ell_t\)（有效载荷 \(m_{\mathrm{eff},t}=\hat\ell_t\hat m_t\)）；**动力学头** \(f_{\mathrm{dyn}}\) → \(z_{\mathrm{dyn},t}\)。两路均有 **\(L_1\) 稀疏** 与物体侧 **质量/装载监督**。
- **对 wiki 的映射：**
  - [SplitAdapter 论文实体](../../wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)（冻结基线：PhysHSI 类 AMP 搬箱策略，见论文 Sec. 3.1）

### 3) 分裂世界模型 + GRL 交叉对抗（Sec. 3.3–3.4）

- **链接：** <https://arxiv.org/html/2606.03297#S3>
- **核心贡献：** **物体世界模型** \(W_{\mathrm{obj}}\) 预测物体位姿/朝向增量 \(\Delta p^{\mathrm{obj}}, \Delta R^{\mathrm{obj}}\)（条件含 \(z_{\mathrm{obj}}\) 与载荷估计）；**动力学世界模型** \(W_{\mathrm{dyn}}\) 预测下一机器人状态。世界模型均条件于 \(m_{\mathrm{est}}\)，鼓励 latent 分工。另设 **GRL 对抗头**：禁止 \(z_{\mathrm{dyn}}\) 预测物体转移、禁止 \(z_{\mathrm{obj}}\) 预测机器人转移，用 **predictability gap**（匹配分支 \(R^2\) − 交叉分支 \(R^2\)）量化解耦；消融显示 **无 split latent** 在 6 kg 全流程掉点最明显，**无 GRL** 次之。
- **对 wiki 的映射：**
  - [SplitAdapter 论文实体](../../wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)（Related Work：AnyAdapter / WMR 等预测式适配同族）

### 4) 分层 FiLM 与训练接口（Sec. 3.5）

- **链接：** <https://arxiv.org/html/2606.03297#S3.SS5>
- **核心贡献：** 对冻结策略中间特征做 **FiLM**：\(h'=\gamma(z)\odot h+\beta(z)\)。**物体/负载 FiLM**（含 \(m_{\mathrm{est}}\)）调制 **较浅层**，塑造抬升姿态与全身协调；**动力学 FiLM** 调制 **靠近动作头** 的深层，补偿 sim–real 动力学差。调制层 **初始化为恒等** 以保留冻结策略行为；适配器与基策略 **共用 RL 目标与奖励** 继续训练。
- **对 wiki 的映射：**
  - [SplitAdapter 论文实体](../../wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)

### 5) 仿真与真机实验（Sec. 4）

- **链接：** <https://arxiv.org/html/2606.03297#S4>
- **核心贡献：** **Isaac Gym 训练** 基策略与适配器，**MuJoCo sim-to-sim** 评测。对比：**PhysHSI 基策略**、**AnyAdapter 式统一 WM-FiLM**、以及 **无 split / 无分层 FiLM / 无 GRL** 变体。条件：**2/4/6 kg** × **0/30/60 cm** 搬放高度，每格 **10 次** 试验，报 **Lift-up** 与 **Full-task**（接近–抬起–搬运–放置全流程）。表 1：SplitAdapter **86/90 Full-task** vs 基线 **71/90**、WM-FiLM **75/90**；**6 kg, 0 cm 地面搬起** 最难。真机 **G1 零样本**：9 条件×3 次，SplitAdapter **26/27 (96.3%)** Full-task、**27/27 Lift-up** vs 基线 **16/27 (59.3%)** / **22/27**；泛化含 **训练外 6 kg**、大箱/亚克力箱、人机递接等（项目页视频）。
- **对 wiki 的映射：**
  - [SplitAdapter 论文实体](../../wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

## 其他公开资料（非 PDF 正文）

- **项目页：** <https://splitadapter.github.io/> — 归档见 [sources/sites/splitadapter-github-io.md](../sites/splitadapter-github-io.md)
- **PhysHSI 基线论文：** Wang et al., <https://arxiv.org/abs/2510.11072> — 知识库策展页 `wiki/entities/paper-amp-survey-15-physhsi.md`

## 当前提炼状态

- [x] 论文摘要与核心方法摘录（≥4 条）
- [x] wiki 页面映射
