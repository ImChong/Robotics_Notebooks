# smpc2rl_arxiv_2608_12063

> 来源归档（ingest）

- **标题：** Learning Loco-Manipulation From SMPC Demonstrations With Sparse Offline-to-Online RL
- **短名：** SMPC-to-RL / SMPC2RL
- **类型：** paper
- **来源：** arXiv abs / PDF / HTML
- **原始链接：**
  - <https://arxiv.org/abs/2608.12063>
  - <https://arxiv.org/pdf/2608.12063>
  - <https://arxiv.org/html/2608.12063>
- **项目页：** <https://pages.rai-inst.com/smpc2rl/> — 归档见 [`sources/sites/rai-inst-smpc2rl.md`](../sites/rai-inst-smpc2rl.md)
- **作者：** Martin Schuck<sup>1,2</sup>, Maks Sorokin<sup>1</sup>, Simone Manni<sup>1,3</sup>, Duy Ta<sup>1</sup>, Angela P. Schoellig<sup>2</sup>, Marco Hutter<sup>1,3</sup>, Simon Le Cleac'h<sup>1</sup>, Jan Brüdigam<sup>1</sup>
- **机构：** RAI Institute；慕尼黑工业大学（TUM）；苏黎世联邦理工（ETH Zurich）
- **版本：** arXiv:2608.12063 v1（2026-08-12）；通读所用 PDF 为 **v2（2026-10-26）**，首页标注 CoRL 2026（10th Conference on Robot Learning, Austin TX）
- **入库日期：** 2026-08-14
- **再核日期：** 2026-11-12（通读 v2 全文并重核项目页）
- **核对范围：** 通读 arXiv v2 PDF 全部 16 页正文文字（摘要、§1–§6、参考文献、附录 A 扩展消融、附录 B 训练细节与表 3、附录 C SMPC 采数与表 4 / 算法 1、附录 D 硬件）；图 4–11 的曲线仅按图注与坐标轴文字读取，未逐点读数；未看项目页视频；论文未附其他补充材料；未运行任何代码（论文无官方仓）。
- **通读：** 已完成
- **一句话说明：** 在仿真里用 **SMPC** 当可交互调参的专家数据机，再以 **稀疏奖励** 做 offline-to-online **FastTD3**；高层策略接冻结 **ReLIC** 低层全身稳定。Spot（带臂）与 **G1** 真机部署，Spot 四任务真机成功率见表 1。

## 核心摘录

### 1) 问题（§1）
- 作者称：把纯 RL 扩展到复杂 loco-manipulation 受限于稠密奖励 shaping，改一次奖励就要一轮长训练；SMPC 好调但计算开销高，难以直接上高自由度硬件（§1）。
- 例子：带臂四足滚大轮胎，物体太重拖不动，必须边踩边推（§1）。
- 遥操作与人体动作重定向对非拟人平台扩展性差（§3）。

### 2) 方法要点（§3、附录 B/C）
1. **分层（§3.1）：** 高层输出增量命令 \(a_{\mathrm{high}}=[\Delta v_{\mathrm{cmd}}\in\mathbb R^3,\Delta q^{\mathrm{arm}}_{\mathrm{cmd}}\in\mathbb R^{n_a},\Delta h_{\mathrm{cmd}},\Delta p_{\mathrm{cmd}}]\)（平面基座速度增量、臂关节目标增量、躯干高度与俯仰增量），Spot 不命令身高与俯仰；低层是按 ReLIC 框架（引文 [4]，Zhu 等 CoRL 2025）训练的全身机动控制器，**训练后冻结**，输出 \(q_{\mathrm{target}}=[q_{\mathrm{legs}},q_{\mathrm{arm}}]\)。数据采集与部署共用此结构，SMPC 与 RL 可互换为高层。
2. **稀疏奖励（§3.2 式 1）：** 到目标 0；crash \(-2/(1-\gamma)\)；否则 \(-1\)；\(\gamma=0.99\)。crash 判据为躯干高度或倾角超出预设限值。
3. **O2O（§3.2）：** 修改版 FastTD3；训练初期 replay buffer 中 **50%** 的转移换成预采专家数据；经验成功率达到阈值后课程式撤掉专家数据，转纯在线（阈值 \(\eta=0.1\)，附录 B 表 3；50% 参考 Ball 等 [10]，附录 A）。
4. **增量动作 + 有界 critic（§3.2、附录 B.1）：** 增量参数化便于在动作空间限最大加速度与速度；critic 末层线性后加 `tanh` 再仿射到理论区间，最好回报 0，最低 Q 为 \(-2/(1-\gamma)\)（崩溃惩罚立即终止回合，不累积后续奖励）；另在 tanh 前加常数偏置使初始输出靠近 0。奖励缩放 0.01 使价值预测靠近 0。
5. **SMPC 专家（§3.3、附录 C、表 4）：** 类似 Howell 等 predictive sampling [23]，随机采样 + warm-start，论文称不需要 MPPI；动作按 \(N_c=10\) 个样条控制点插值到地平线 \(T=50\)，\(f=50\) Hz，每次只执行 \(N_e=10\) 步，即 5 Hz 策略、50 Hz 动作块；每 tile \(K=256\) 条样本，噪声 \(\sigma\in[0.5,1.0]\)，\(\gamma=0.99\) 折现 SMPC 代价；缓存仿真状态、按最优样本重置；失败回合从数据集丢弃（算法 1）。
6. **速度（§3.3）：** 单张 RTX 5090 上 SMPC 约 0.5× 实时；约 100 万样本/小时；「在单 GPU 四小时内」足以引导稀疏 RL。附录 C 给出各任务分块采数耗时：reach 1:39h、box push 2:41h、tire upright 3:28h、tire roll 3:15h、G1 10:06h。
7. **训练（附录 B）：** TD3 + FastTD3 设置，MuJoCo Warp + mjlab；物体质量 / 尺寸 / 摩擦随机；非对称 actor-critic（actor 噪声观测、critic 精确观测）；RL 训练 20–90 分钟（RTX 5090）；G1 动作正则 \(8\times10^{-4}\)、critic lr \(5\times10^{-4}\)；G1 上 ReLIC 做了适配并保留躯干倾角与高度控制。
8. **SMPC 调参实况（附录 C）：** 从距离目标奖励与摔倒惩罚起，逐步加夹爪-物体奖励、躯干 / 腿触碰物体惩罚、动作平滑等；滑块实时调；每任务调参 <1 小时；一次实时调参回合 box push 10.1 s、tire roll 14.5 s；作者估计看 100 组参数 <30 分钟，而 RL 需 >100 小时。投稿视频里的多模态行为来自缺少接触惩罚的中间调参版本。

### 3) 实验（§4、附录 A）
- **任务（图 2）：** Spot reach、box push、tire upright、tire roll；G1 box push。图 4：仿真成功率近 100%，5 seed；无 SMPC 数据时「什么也学不到」。
- **真机（§4 表 1，Spot，十次运行）：** reach 10/10、box push 9/10、tire upright 10/10、tire roll 8/10。G1 真机论文只定性说可部署，**未给成功率**。
- **Q1（图 5）：** 策略任务时长一致短于 SMPC；部分任务改进 >50%；任务时长标准差降 11–45%；作者将 SMPC 方差归因于并行环境数有限导致次优轨迹。SMPC 统计为完整 4M 样本数据集，策略统计为 50k 个无探索噪声仿真回合。
- **Q2（图 6）：** 数据量从数十万到数百万；简单任务不敏感，复杂任务需要更大数据集；最难任务需约 400 万样本（4 GPU 小时内采集）收敛。作者「posit」原因为 critic 梯度估计需要更广的状态覆盖。
- **Q3（图 7）：** tile 内采样环境数 64 / 128 / 256；多数任务稳健，tire roll 掉点。
- **Q4（图 8）：** 多模态 SMPC 数据（踢轮胎、肩推、踩进轮胎）使 Spot tire roll 策略完全学不起来，尽管演示成功率略高；通过更严的代价项（腿 / 身体接触惩罚）把行为收成单模态。
- **Q5（表 2，仿真，5 次运行 × 各 100 回合，放宽最长回合）：** FastTD3 用 SMPC 稠密代价、IQL 用 SMPC 数据与稠密代价、DP（Diffusion Policy）用 SMPC 数据。reach 三者都 1.00；box push 0.00 / 0.03 / 0.10；tire upright 0.00 / 0.01 / 0.01；tire roll 0.02 / 0.04 / 0.01；本方法 0.98±0.03、0.99±0.005、1.00±0、0.98±0.02。作者称这是 SMPC 代价与 RL 不匹配而非算法本身的局限，并称 DAgger 式设置因运行 SMPC 过贵而难以接入。
- **附录 A：** 专家比例（0.1 / 0.25 / 0.5 / 0.75 / 0.9）：reach、box、upright 几乎不受影响，tire roll 低于 50% 无法收敛，更高比例在撤出后仍更稳（作者假设为 critic 在成功态附近估值更准）。有界 critic：critic lr 提到 \(8\times10^{-4}\) 时无界 critic 出现性能尖峰掉点（Spot box push），有界更稳。撤出阈值（0.1 / 0.25 / 0.5 / 1.0）：长期保留 SMPC 数据在除 reach 外所有任务上显著退化，高于 25% 变慢，tire upright 与 tire roll 阈值 ≥50% 在步数上限内不收敛；作者解释为 SMPC 数据离当前策略太远，成了干扰。

### 4) 局限（§6，作者自述）
- 最优性是局部的，策略绑在 SMPC 数据分布附近，不期望发现根本不同的全局策略；超参（网络尺寸、学习率）影响结果。
- 低层冻结，难适应任务级物理扰动（通用低层训练时臂上加了较大力扰动）；后期在线阶段解冻是方向。
- 真机部署依赖状态信息；非结构化 / 户外需视觉数据或蒸馏。

### 5) 开源核查（步骤 2.5）
- **项目页（2026-11-12 重核）：** 外链仅含论文、`static/paper.pdf`、judo（<https://github.com/rai-opensource/judo>）、两篇 arXiv 引用与模板致谢；**无本文训练 / 部署仓**。judo 是通用采样 MPC 工具箱，不是本文管线。
- 论文 PDF 未列代码链接，仅给项目页。
- **结论：** 确认未开源（截至 2026-11-12）。故 wiki 页 `code:` 不填。

## FastTD3 关键超参（附录 B 表 3）

| 参数 | 值 |
|------|----|
| Buffer \(B\) | \(2^{24}=16\,777\,216\) |
| 每步梯度 \(N_\nabla\) | 8 |
| 撤出阈值 \(\eta\) | 0.1 |
| Batch \(N_b\) | 8192 |
| \(\gamma\) / Polyak \(\tau\) | 0.99 / 0.005 |
| Actor / Critic lr | \(3\times10^{-4}\) / \(1\times10^{-4}\)（G1 critic \(5\times10^{-4}\)） |
| Policy delay / 平滑噪声 | 2 / 0.05 |
| 动作正则 | \(5\times10^{-4}\)（G1 \(8\times10^{-4}\)） |
| Reward scale | 0.01 |

## 对 wiki 的映射

- 升格 [SMPC-to-RL 论文实体](../../wiki/entities/paper-smpc2rl-loco-manipulation.md)
- 更新 [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Sumo](../../wiki/methods/sumo.md)、[MPC vs RL](../../wiki/comparisons/mpc-vs-rl.md)、[Online vs Offline RL](../../wiki/comparisons/online-vs-offline-rl.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[mjlab](../../wiki/entities/mjlab.md)、[Model Predictive Control](../../wiki/methods/model-predictive-control.md)

## 当前提炼状态

- [x] 摘要 + 分层 + 稀疏奖励 + 附录超参 / 采数 / 消融 + 开源边界
- [x] 通读 v2 全文并补表 1 真机成功率、表 2 基线对比、采数耗时
- [x] wiki 实体页与交叉引用
- [x] `sources/sites/`（无官方仓）+ `sources/repos/judo.md`（对照仓）
