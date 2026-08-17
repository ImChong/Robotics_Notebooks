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
- **版本：** arXiv:2608.12063（2026-08-12）；Under submission
- **入库日期：** 2026-08-14
- **再核日期：** 2026-08-17
- **一句话说明：** 在仿真里用 **SMPC** 当可交互调参的专家数据机，再以 **稀疏奖励** 做 offline-to-online **FastTD3**；高层策略接冻结 **ReLIC** 低层全身稳定。Spot（带臂）与 **G1** 真机推箱/扶胎/滚胎。项目页截至再核日 **仍未列 GitHub**。

## 核心摘录

### 1) 问题
- 复杂 loco-manipulation 的 on-policy RL 依赖 **稠密奖励 shaping**；改一项奖励就要再训一轮，迭代极慢。
- 现成模仿/轨迹跟踪把策略绑死在演示质量上；遥操作与人体重定向对 **非人形**（带臂 Spot）也不好扩。
- **SMPC** 在仿真里可近实时调代价（RTX 5090 约 **0.5×** 实时），但不适合直接上高 DoF 真机。

### 2) 方法要点
1. **分层：** 高层任务策略输出增量命令 \(a_{\mathrm{high}}=[\Delta v_{\mathrm{cmd}},\Delta q_{\mathrm{cmd}}^{\mathrm{arm}},\Delta h_{\mathrm{cmd}},\Delta p_{\mathrm{cmd}}]\)（Spot 固定身高/俯仰）；低层是冻结的 **ReLIC** 全身机动策略，出全关节位置目标并保平衡。采数与部署共用同一接口，SMPC 与 RL 可互换高层。
2. **稀疏奖励：** \(r=0\) 到目标，\(r=-2/(1-\gamma)\) 摔倒（\(\gamma=0.99\)），否则 \(r=-1\)。摔倒阈值取躯干高度/倾角。无穷地给 \(-1\) 的折现和是 \(-1/(1-\gamma)\)，摔倒罚两倍，避免学成自杀捷径。实现时再乘 **reward scale 0.01**，把 Q 压近 0。
3. **Offline-to-online FastTD3：** 早期 replay **50%** 替换为 SMPC 专家转移；经验成功率过 **\(\eta=0.1\)** 后 curriculum 撤掉专家数据，转纯在线。
4. **增量动作 + 有界 critic：** 增量参数化可在动作空间硬限加速度/速度；稀疏回报有已知上下界（最好 0、最差 \(-2/(1-\gamma)\)），critic 末层 `tanh` 后仿射到该区间。
5. **SMPC 采数（附录 C）：** 随机采样 + warm-start（不必上 MPPI）；动作按 **样条控制点** 采样再插值，避免逐步独立噪声。\(T=50\)、\(N_c=10\)、\(f=50\,\mathrm{Hz}\)、每次只执行 \(N_e=10\) 步 → **5 Hz 规划 / 50 Hz 动作块**。每 tile \(K=256\) 条采样；elite 状态 checkpoint 后广播。约 **100 万样本/小时**；最难任务约 **400 万样本 / 4 GPU 小时**。失败回合从 buffer 丢弃。
6. **Sim2Real：** 物体质量/尺寸/摩擦随机；非对称 actor-critic（actor 噪声观测、critic 真状态）。栈：MuJoCo Warp + mjlab。G1 动作正则 \(8\times10^{-4}\)、critic lr \(5\times10^{-4}\)（Spot 默认 \(5\times10^{-4}\) / \(1\times10^{-4}\)）。

### 3) 实验（论文报告摘要）
- **五任务：** Spot reach / 推箱 / 扶胎直立 / 滚胎；G1 推箱。仿真近 100% 成功率（5 seed）；无专家数据则稀疏奖励学不动。
- **Q1 超越教师：** 稀疏策略任务时间 consistently 快于 SMPC；部分任务 **>50%**；时长标准差降 **11–45%**。教师统计覆盖完整 4M 数据集，策略统计为 50k 无探索噪声仿真回合。
- **Q2 数据量：** 简单导航对样本不敏感；滚胎等协调任务需要约 **4M**。
- **Q3 质量：** 减少 tile 内采样环境大多任务仍稳；滚胎明显掉。
- **Q4 多模态：** 多模态 SMPC（踢/肩推/踩进轮胎）即使演示成功率更高，**uni-modal 策略完全学崩**；必须用更严代价把行为收成单模态。
- **附录 A：** 专家比例大多任务不敏感，滚胎 **低于 50% 会塌**；有界 critic 在激进 lr（\(8\times10^{-4}\)）下避免尖峰掉点；撤出阈值高于 ~10–25% 会拖慢几乎所有任务，**一直留专家数据几乎全变差**。
- **真机：** 五任务两本体均可部署；无平台专用奖励工程。Spot 策略机外推理、WiFi 下发；G1 机载。状态用 **OptiTrack** + 本体传感。箱子 \(0.5^3\,\mathrm{m}\)、1.2 kg；轮胎半径 0.33 m、宽 0.34 m、**14.3 kg**。

### 4) 局限
- 行为最优是 **局部的**：策略绑在 SMPC 数据流形附近，不期望发现完全分叉的全局最优。
- 低层冻结 → 难适应任务级扰动；后期解冻是开放方向。
- 部署靠状态量；非结构化/户外需视觉蒸馏。
- 网络宽度与学习率仍影响结果；有界 critic 只是把可用超参区间拉宽。

### 5) 开源核查（步骤 2.5）
- **项目页（再核 2026-08-17）：** 论文、概述视频、方法叙事、消融 tab、BibTeX。相关链接指向 **judo**（通用采样 MPC 工具箱），**不是** 本文训练/部署仓。无 Code 按钮。
- **GitHub：** 项目页与 PDF **未列** 官方仓库；检索 `smpc2rl` 无匹配官方实现。
- **对照仓：** [`sources/repos/judo.md`](../repos/judo.md) — <https://github.com/rai-opensource/judo>（MIT，PyPI `judo-rai`）。可交互调采样 MPC，**不能**复现 tiled 采数 + FastTD3 + ReLIC。
- **结论：** **确认未开源**（截至 2026-08-17）。wiki `## 源码运行时序图` 标不适用。勿把 judo 写成可复现本文管线。

## FastTD3 关键超参（附录 B 表 1）

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

- [x] 摘要 + 分层 + 稀疏奖励 + 附录超参/采数/消融 + 开源边界
- [x] wiki 实体页与交叉引用
- [x] `sources/sites/`（无官方仓）+ `sources/repos/judo.md`（对照仓）
