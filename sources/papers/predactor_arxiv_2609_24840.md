# PredActor: Predictive Action Diffusion for Steerable Onboard Humanoid Control（arXiv:2609.24840）

> 来源归档（ingest）

- **标题：** PredActor: Predictive Action Diffusion for Steerable Onboard Humanoid Control
- **类型：** paper / humanoid / diffusion-policy / onboard-control
- **arXiv abs：** <https://arxiv.org/abs/2609.24840>
- **PDF：** <https://arxiv.org/pdf/2609.24840>
- **项目页：** <https://masteryip.github.io/predactor.github.io/> — 归档见 [`sources/sites/predactor-masteryip-github-io.md`](../sites/predactor-masteryip-github-io.md)
- **代码：** **主体已开源（仅 DAgger 未开放）** — <https://github.com/MasterYip/PredActor> 已开放 MuJoCo 评测、数据采集与 MotionCLIP 语义染色（`PredActor/collection/`）、离线 BC 训练（`PredActor/train.py`）与 G1 LibTorch C++ / Unitree 机载部署栈（`PredActor/diffusion_deploy/`）；README 仅剩 **DAgger** 未勾选。评测/部署工件见 [PredActor_Artifacts](https://huggingface.co/MasterYip/PredActor_Artifacts)（[`sources/repos/predactor-artifacts.md`](../repos/predactor-artifacts.md) 与 [`sources/repos/predactor.md`](../repos/predactor.md)）。
- **机构：** 哈尔滨工业大学（Harbin Institute of Technology）、上海创新研究院（Shanghai Innovation Institute）、RoboParty Lab、清华大学（Tsinghua）、上海交通大学（SJTU）等
- **入库日期：** 2026-09-24
- **复核日期：** 2026-11-19
- **核对范围：** 通读 `arXiv:2609.24840v3` 全文（正文 §I–§VII、Table 1–4、Fig. 1–7 与 Appendix A–E / Table 8），并核对公开仓库 `MasterYip/PredActor` 的当前发布清单
- **通读：** 已完成
- **一句话说明：** 联合状态–动作扩散人形控制器：仅以本体感知历史与可选任务条件为输入，内部共同去噪未来动作与未来运动学状态以支持免重训 CG（速度/落点/全身）与文本 CFG，并通过异步非同步扩散调度与两步 NS-DDIM 在 Unitree G1 机载 Jetson Orin NX 上实现 50 Hz 闭环。

## 核心摘录（面向 wiki 编译）

### 1) 范式与核心机制

- **联合预测（Predictive Action Diffusion）：** 策略同时扩散未来 20 步状态轨迹 $\hat{\mathbf{s}}_{t:t+H}$ 与动作序列 $\hat{\mathbf{a}}_{t:t+H}$，未来状态完全留在扩散内部表示中，不经独立运动跟踪器（如 ARDY+SONIC）二次重放。
- **双路径可控性（CFG + CG）：**
  - **文本 CFG：** 通过 G1 微调 MotionCLIP 语义嵌入 + 自适应条件 dropout / 任务掩码实现语义运动切换与插值；
  - **测试时物理导向 CG（Whole-Body Guidance, WBG）：** 基于预测的局部根节点/躯干/末端状态残差，通过向量 Jacobian 乘积将 PD 目标误差反传为去噪均值偏移，支持速度、偏航、骨盆高度与世界坐标目标点跟随（destination follow），无需针对导航任务重训模型。
- **工程化实时设计：** 采用异步状态-动作噪声调度、两步确定性非同步 DDIM（`NS-DDIM2`）、滚动去噪队列（rolling denoising）与五层 C++/LibTorch 推理加速，将 actor 前向耗时降至机载 20 ms 周期以内。

### 2) `arXiv:2609.24840v3` 量化实验结果（Table 2, Table 3, Table 4, Table 8）

| 评估维度 | PredActor（v3 实测） | 关键对照（文内） |
|----------|----------------------|------------------|
| 基础平稳性与时延（Table 2 Basic） | Jerk RMS `16.810 ± 1.409` `[60/60]`；推理延迟 `1.321 ± 0.011 ms` `[15/15]` | `CLoC+FK` Jerk `266.605 ± 32.412` / `6.102 ms`；`DiffuseLoco w TextCFG` `15.899` / `12.141 ms` |
| 300 N 外力推扰（Table 2 Perturbation） | 存活率 `0.511 ± 0.030` `[300/300]`（匹配无 CG 变体 `0.579 ± 0.012`） | `CLoC+FK` `0.303`；`DiffuseLoco w TextCFG` `0.563 ± 0.009` |
| 零样本目标点导航（Table 2 Point Nav） | 存活率 `0.983 ± 0.029` `[45/45]`，到达率 `0.978 ± 0.028` `[45/45]`，末态有效误差 `0.605 m`（最小 `0.604 m`） | `CLoC+FK` 到达率 `0.000 ± 0.053`（误差 `3.194 m`）；`ARDY+SONIC` 仅 `[2/15*]` 合法步态 |
| 文本指令检索（Table 2 Text Command） | 存活率 `0.888 ± 0.068` `[135/135]`，MotionCLIP 检索分 `0.539 ± 0.075` `[119/135*]` | `DiffuseLoco` 检索分 `0.424 ± 0.159`；`PredActor w/o state` 跌至 `0.397 ± 0.023` |
| 状态联合预测消融（Table 2 / Table 8） | 去掉状态预测（`w/o state`）使推扰存活降为 `0.487 ± 0.033`、文本检索降至 `0.397 ± 0.023`，且**无法使用状态驱动的点导航 CG** | 证明联合未来状态预测既支撑免重训 CG 也反哺本体与文本语义对齐 |
| 机载 Jetson Orin NX 实时性（Table 4） | TorchScript + LibTorch `inference_mode`（`NS-DDIM2`, B=1 FP32）：Actor p50/p95 `13.175 / 13.400 ms`；**完整 live callback p50/p95 `16.790 / 19.383 ms`**，**593/600 次满足 20 ms（50 Hz）预算** | 相对结构基线 `21.461 / 22.036 ms`（600/600 超时）实现逐级加速 |
| Unitree G1 真机试验（Table 3, 5 trials/组） | `Stand→Walk` `5/5`、`Walk→Jog` `5/5`、`Stand→Squat` `5/5`、`Squat→Stand` `4/5`、`Stand push` `4/5`、`Walk push` `4/5` | 实机验证 50 Hz 机载站立抗扰与步态无缝切换 |

> 注：早期入库摘录（2026-09-24）记录的 `15/15`、`0.580`、`0.535` 为旧预印本口径，2026-11-19 已全面校订为 `arXiv:2609.24840v3` 终稿数据。

## 开源状态演变核查（2026-11-19）

- **2026-10-03（占位） → 2026-10-08（仅 MuJoCo 评测） → 2026-11-19（主体发布）：** 官方仓库 `README.md` Release Checklist 已勾选 `[x]` Evaluation、`[x]` Data collection & MotionCLIP dyeing、`[x]` BC training 与 `[x]` Deployment（`diffusion_deploy`），目前仅 **`[ ] DAgger`** 尚未公开。

## 对 wiki 的映射

- 复用并升级：[paper-predactor](../../wiki/entities/paper-predactor.md)
- 交叉：[diffusion-policy](../../wiki/methods/diffusion-policy.md)、[sonic-motion-tracking](../../wiki/entities/nvidia-sonic.md)、[whole-body-control](../../wiki/concepts/whole-body-control.md)、[locomotion](../../wiki/tasks/locomotion.md)、[humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md)
