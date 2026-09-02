# RISE: Adaptive Imagination for World Action Models（arXiv:2608.20430）

> 来源归档（ingest）

- **标题：** RISE: Adaptive Imagination for World Action Models
- **缩写：** **RISE**（Refining Imagination through SElective Rollout）
- **类型：** paper / world-action-models / autonomous-driving / latent-imagination
- **arXiv：** <https://arxiv.org/abs/2608.20430>（Submitted 2026-08-20；HTML：<https://arxiv.org/html/2608.20430>；PDF：<https://arxiv.org/pdf/2608.20430>）
- **项目页：** <https://cowarobot-ai.github.io/RISE/> — 归档见 [`sources/sites/cowarobot-ai-github-io-rise.md`](../sites/cowarobot-ai-github-io-rise.md)
- **代码：** <https://github.com/COOWAI/RISE>（MIT）— 归档见 [`sources/repos/coowai-rise.md`](../repos/coowai-rise.md)
- **数据集：** <https://huggingface.co/datasets/COWARobot/CounterDrive>（CC-BY-NC-ND-4.0，约 31.5 GB）— 归档见 [`sources/datasets/counterdrive.md`](../datasets/counterdrive.md)
- **作者：** Hongbo Lu\*（COWARobot / 上海交通大学）、Liang Yao\*（COWARobot / 河海大学）、Chenghao He\*、Hao Han\*、Fan Liu（河海大学）、Wenlong Liao†、Tao He、Pai Peng†‡（均为 COWARobot，除非另标）
- **机构：** 酷哇科技（COWARobot Co. Ltd）/ 上海交通大学 / 河海大学
- **入库日期：** 2026-09-02
- **一句话说明：** 给驾驶 World Action Model 加可插拔 Scheduler：用 Latent Evaluator 估计已揭示风险与 Future Planning Gain，用 Rollout Gate 逐步做 Roll/Stop；配套反事实数据集 CounterDrive。**不是** OpenDriveLab 那篇同名 RISE（arXiv:2602.11075）。

## 开源状态（步骤 2.5，2026-09-02）

| 资源 | 状态 | 说明 |
|------|------|------|
| 项目页 | 已上线 | 摘要、CounterDrive 规模、NAVSIM/nuScenes 表、BibTeX `2608.20430`；链 GitHub / HF / 前作 DAWN |
| 代码 | **已开源（MIT）** | [COOWAI/RISE](https://github.com/COOWAI/RISE)：训练 YAML、`app.main` 七段手动链、Oracle / EPDMS CLI；**无官方权重** |
| 数据集 | **已发布** | HF `COWARobot/CounterDrive`：`ConterDrive-NAVSIM.tar` 20.2 GB + `ConterDrive-nuScenes.tar` 11.3 GB；README 几乎空 |
| 权重 | **未发布** | README 仍写 “weights … are not released” |
| README 滞后 | 注意 | GitHub README（约 2026-08-13）仍写论文/数值/反事实数据未发布；**以项目页 + arXiv + HF 实际链接为准** |

**结论：已开源（训练/评测代码 + CounterDrive 数据）；权重未发，完整复现还需自备 NAVSIM/NuPlan 资产与自训 checkpoint。**

## 摘录 1：问题与主张（§1）

- **痛点：** 现有 WAM 把想象预算做成全局超参——imagine-and-plan / imagine-then-plan / 无想象，都不会在每步前缀后问「再滚一步还值不值」。
- **信号：** Future Planning Gain \(B_h\)：相对当前前缀停止，继续滚到各合法深度时规划分的变化；应逐步重估，而不是事先选定最终 horizon。
- **方法：** RISE 在 Encoder–Predictor–Planner 上插 **Scheduler**（Latent Evaluator + Rollout Gate）。Evaluator 出 Risk Profile \(R_h\) 与 Gain Profile \(B_h\)；Gate 用代价偏好 \(\lambda\) 出 \(x_h\)，\(x_h>0\) 则 Roll，否则 Stop。有效深度 \(K(c;\lambda)\in\{0,\ldots,H\}\) 由序列决策涌现。
- **数据缺口：** 事实日志只有一条实现未来。CounterDrive 用反事实视频补多样结局与局部风险监督。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-rise-adaptive-imagination-wam.md`](../../wiki/entities/paper-rise-adaptive-imagination-wam.md)；与 [World Action Models](../../wiki/concepts/world-action-models.md)、[Latent Imagination](../../wiki/concepts/latent-imagination.md) 互链。同名对照 [OpenDriveLab RISE](../../wiki/entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md)。

## 摘录 2：CounterDrive（§3）

- 从 NAVSIM / nuScenes **选中场景**生成 10 s / 1080p 反事实视频（Wan 2.7，2 Hz → 20 帧），prompt 钉死相机、道路与初始交通。
- OpenVO 恢复 ego 位姿 \(\tilde{\tau}\) 与相邻动作 \(\tilde{a}\)。
- 人工核验：ego 运动一致性、事故起始帧、生成畸变、因果类（正常 / 非自车 / 自车）；自车事故另记建议避让。严重畸变或轨迹不可靠则丢弃。
- 规模：**NAVSIM 5013/1000**、**nuScenes 2432/511**（train/test）。配对只覆盖所选子集，**不是**原训练集一一对应。
- 接受片段监督未来预测；核实后的事实–反事实对提供时序局部 Risk Profile 排序。

**对 wiki 的映射：** 数据集归档 [`sources/datasets/counterdrive.md`](../datasets/counterdrive.md)；实体页写清「安全评测 AUC 从随机升到 0.93–0.96」与覆盖缺口。

## 摘录 3：架构与三阶段训练（§4）

| 模块 | 做法 |
|------|------|
| Encoder | 冻结 V-JEPA 2 ViT-L；4 帧 \(256\times512\)，\(16\times16\) patch + 两帧 tubelet → 2 个观测 latent 步 × 512 token × 1024-d |
| Predictor | 12 层、384-d、12 头、自车相对位姿条件、帧因果 Transformer；最多 \(H_{\mathrm{NAVSIM}}=4\) / \(H_{\mathrm{nuScenes}}=3\) |
| Planner | 12 层扩散 Transformer；20 步 DPM-Solver++；6 条候选，每条 8 个 \(0.5\) s 位姿 \((x,y,\cos\psi,\sin\psi)\) |
| Evaluator | 空间池化 + 1 层因果 GRU（512-d）→ \(R_h,B_h\) |
| Gate | LayerNorm + 两层 128-d FC；\(c_h=h\)，默认 \(\lambda=0.005\) |

三阶段：I 训 Predictor + 变前缀 \(\Pi_0\)；II 学风险/增益并 refinement 后训 \(\Pi_1\)；III 冻结其余、用全 horizon 效用曲线蒸馏 Gate。Stop 且 \(h>0\) 时对前缀做两步风险梯度 refinement（步长 0.05，范数帽 0.25）。

**对 wiki 的映射：** 实体页画流程总览 + 对齐仓库 `P0 → Field → Calibration → P1 → Stop → Oracle → Gate` 的源码时序图；骨干对照 [V-JEPA 2](../../wiki/entities/paper-vjepa2.md)。

## 摘录 4：评测与消融（§5）

| 基准 | RISE | 相对最强基线 |
|------|------|----------------|
| NAVSIM v1 PDMS | **91.5** | DriveFuture 90.7（+0.8）；EP **98.3** / TTC **98.6** |
| NAVSIM v2 EPDMS | **90.8** | DriveFuture 89.9（+0.9）；九项中七项第一或并列 |
| nuScenes Avg. L2 / 碰撞 | **0.31 m / 0.10** | DAWN 0.33 / 0.11 |

- Scheduler ⊕ CounterDrive：88.9/89.7 → **90.8/91.5**（EPDMS/PDMS）；二者单独也涨，合用最好。
- 自适应对照：Random Stop 2.03 步 / 89.5；Latent Margin 2.98 步 / 89.7；**Scheduler 2.40 步 / 287 ms / 90.8**。
- 固定深度不是全局最优：1248 景 \(h^*=0\)（加深反而掉分），4036 / 2180 景偏好 \(h=3/4\)。
- 反事实测试集：无 CounterDrive 时 AUC≈0.49–0.52、事故 Acc 0.51；有则 AUC **0.93–0.96**、Acc **0.96**。
- 插到前作 DAWN（不改其 Predictor/Planner）：PDMS **89.1→90.3**。

**对 wiki 的映射：** 实体页用「规划分 vs 平均 rollout」读法；驾驶域对照 [X-Foresight](../../wiki/entities/paper-x-foresight.md)、索引级 [Latent-WAM](../../wiki/entities/paper-sa-2603-24581-latent-wam-latent-world-action-modeling-for-end.md)。

## 建议 wiki 动作

- 新建深度论文实体 [`wiki/entities/paper-rise-adaptive-imagination-wam.md`](../../wiki/entities/paper-rise-adaptive-imagination-wam.md)
- 交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[Latent Imagination](../../wiki/concepts/latent-imagination.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)、[V-JEPA 2](../../wiki/entities/paper-vjepa2.md)、[X-Foresight](../../wiki/entities/paper-x-foresight.md)、同名 [OpenDriveLab RISE](../../wiki/entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md)
- 不更新 `index.md`（非核心入口变化）
