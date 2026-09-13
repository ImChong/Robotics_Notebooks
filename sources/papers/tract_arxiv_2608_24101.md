# TrAct（arXiv:2608.24101）

> 来源归档（ingest）

- **标题：** TrAct: Bridging Robot Control and Visual Prediction with Visual Tracks
- **短名：** TrAct（Track & Act）
- **类型：** paper / manipulation / world-model / vla / visual-tracks / closed-loop
- **arXiv：** <https://arxiv.org/abs/2608.24101>
- **PDF：** <https://arxiv.org/pdf/2608.24101>
- **HTML：** <https://arxiv.org/html/2608.24101>
- **alphaXiv：** <https://www.alphaxiv.org/abs/2608.24101>
- **项目页：** 无独立项目站（截至入库日仅 arXiv 与第三方索引）
- **代码：** 论文写明 *Code, data, and trajectories will be released*；检索未见官方 GitHub / Hugging Face → **宣称将开源 / 待发布**
- **作者：** Zhi Cao、Howard Ji、Kevin Zhang、Kuangzhi Ge、Li Fei-Fei、Jiajun Wu、Huang Huang（* 共同一作 / 共同指导）
- **机构：** 密歇根大学（UMich）；斯坦福大学（Stanford）
- **版本：** arXiv:2608.24101（2026-08）
- **入库日期：** 2026-09-13
- **一句话说明：** 用 2D 视觉轨迹作控制与预测之间的具身无关中间接口：VLAT 在 π₀.₅ 上联合预测动作–轨迹对，轨迹条件世界模型（TWM）滚未来视频，VLAC 选最优 rollout 再执行；LIBERO-INTEGRAL 27%→55%，真机 Franka 49%→76%。

## 摘要级要点

- **问题：** 机器人动作低维且具身相关，与图像空间视觉变化弱对齐；动作条件世界模型需从稀疏命令推断稠密像素未来，易幻觉成功。
- **主张：** 2D 点轨迹描述任务相关点在图像中的运动，跨具身共享、对视频预测提供稠密空间引导。
- **三模块：** (1) **VLAT** — 在 flow-matching VLA π₀.₅ 上改动作头，联合输出 K 组动作块与 2D 轨迹；(2) **TWM** — SVD + ControlNet，以轨迹渲染为空间控制图条件生成未来视频；(3) **VLAC** — InternVL2 奖励模型对想象 rollout 打分选优。
- **轨迹监督：** 7 个夹爪 mesh 顶点 + 腕部视角 5×5 背景网格点；背景用 CoTracker 跟踪；agent 视角仅夹爪点，wrist 视角含夹爪+场景点。
- **预训练：** VLAT 在 76K DROID + 150K EgoDex（1:2）上 30K 步；TWM/AWM 在 76K DROID 上 30K 步；4×H100，batch 64。
- **基准 LIBERO-INTEGRAL：** 20 任务 = 10 鲁棒性（LIBERO-PRO 物体/位置/任务 + LIBERO-Plus 相机/初始化）+ 10 跨具身（Franka→UR5）；相对标准 LIBERO 更能测分布与 embodiment shift。
- **开源（截至 2026-09-13）：** 论文承诺发布代码/数据/轨迹；无项目页、无官方仓库或 HF 权重 → **待发布**。未建 `sources/repos/` / `sources/sites/`。

## 核心摘录（面向 wiki 编译）

### 1) VLAT 联合 flow-matching 目标

\[
\pi_{\theta}(o_t,l)\rightarrow\{(a_i,\tau_i)\}_{i=1}^{K},\quad
\mathcal{L}_{\text{VLAT}}=\mathcal{L}_{\text{flow}}(a,a^{*})+\lambda\mathcal{L}_{\text{flow}}(\tau,\tau^{*})
\]

\(a_i\in\mathbb{R}^{H\times d_a}\) 为 16 步动作块；\(\tau_i\in\mathbb{R}^{N\times H\times 2}\) 为 N 个 2D 轨迹。EgoDex 仅监督轨迹头，机器人数据同时监督动作头。

### 2) TWM vs AWM（Table 1 仿真 agent 视角节选）

| 指标 | AWM | TWM |
|------|-----|-----|
| PSNR ↑ | 15.12 | **24.51** |
| SSIM ↑ | 0.482 | **0.843** |
| LPIPS ↓ | 0.438 | **0.106** |
| FVD ↓ | 129 | **38** |

轨迹条件在所有视角与真机/仿真域一致优于动作条件。

### 3) LIBERO-INTEGRAL 成功率（Table 3）

| 方法 | Swap | Object | Task | Camera | RobotInit | Cross-Emb. | **Avg.** |
|------|------|--------|------|--------|-----------|------------|----------|
| π₀.₅ | 0.40 | 0.35 | 0.15 | 0.45 | 0.45 | 0.17 | **0.27** |
| VLAT | 0.50 | 0.40 | 0.35 | 0.45 | 0.55 | 0.42 | 0.44 |
| VLAT+AWM | 0.65 | 0.50 | 0.50 | 0.45 | 0.55 | 0.44 | 0.49 |
| **TrAct** | 0.65 | **0.60** | 0.50 | **0.60** | **0.65** | **0.50** | **0.55** |

三 seed 复现：TrAct mean **0.547** vs VLAT+AWM **0.490**（95% CI 不重叠）。

### 4) 真机 Franka（Table 4 平均）

| 方法 | Avg. SR |
|------|---------|
| π₀.₅ | 0.49 |
| π₀.₅ + VLAC | 0.65 |
| VLAT | 0.55 |
| VLAT+AWM | 0.66 |
| **TrAct** | **0.76** |

未见背景（diff back）下 TrAct 仍 0.7，VLAT+AWM 跌至 0.4。

### 5) 推理与候选数

- 仿真 K=20、真机 K=16；Temperature-Scaled Resampling（noise scale 2，TSR k=3/4）。
- K 消融：5→10→20 成功率 0.52→0.54→0.55，边际递减。

## 开源核查（步骤 2.5）

- 论文 Conclusion 前写明：**Code, data, and trajectories will be released**。
- 无 `*.github.io` / lab 项目页；用户指定 GitHub、Hugging Face **暂未发布**；Robot Papers / CatalyzeX 仅为索引。
- → **宣称将开源 / 待发布**（截至 **2026-09-13**）。后续 lint 可跟进仓库链接。

## 对 wiki 的映射

- 升格 [TrAct 论文实体](../../wiki/entities/paper-tract.md)
- 交叉：[π₀.₅](../../wiki/entities/paper-pi05-open-world-vla.md)、[Ctrl-World](../../wiki/entities/paper-ctrl-world.md)、[SC3-Eval](../../wiki/entities/paper-sc3-eval.md)、[PhysisForcing](../../wiki/entities/paper-physisforcing.md)、[generative-world-models](../../wiki/methods/generative-world-models.md)、[VLA](../../wiki/methods/vla.md)、[manipulation](../../wiki/tasks/manipulation.md)、[world-action-models](../../wiki/concepts/world-action-models.md)

## 当前提炼状态

- [x] 方法、主表、LIBERO-INTEGRAL 基准、开源结论
- [x] wiki 实体与交叉引用
