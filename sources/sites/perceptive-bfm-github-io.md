# Perceptive BFM 项目页（acodedog.github.io/perceptive-bfm）

> 来源归档（ingest 配套站点）

- **URL：** <https://acodedog.github.io/perceptive-bfm/>
- **标题：** Perceptive Behavior Foundation Model: Adapting Human Motion Priors to Robot-Centric Terrain
- **机构：** 妙动科技；香港科技大学（广州）；香港科技大学；中国科学技术大学人工智能研究院
- **入库日期：** 2026-06-11
- **论文：** CoRL 2026 submission（页面标注 *Under review*）；arXiv **TBA**；代码 **TBA**
- **一句话说明：** 官方落地页：提出 **Perceptive BFM**——保留 **原始人体运动参考** 作为部署接口，用 **机器人中心地形感知** 在线补全落脚、间隙、姿态与接触时序；训练期 **TCRS** 离线合成地形一致监督，经 **PMT** 四阶段（盲 teacher → 视觉 student → 目标帧对齐蒸馏 → identity-gated 残差微调）在 **Unitree G1** 上单策略覆盖 locomotion、风格动作、杂技与 mocap 遥操作，并演示楼梯、坡、稀疏支撑、草地与户外真机。

## 页面要点（2026-06 快照）

### 问题：Operator–Environment Mismatch

- 动捕操作员或平地录制的参考传达 **意图与风格**，但不包含机器人所处地形上的 **有效落脚、间隙、身体高度与接触时序**。
- 现有 motion-centric foundation policy 多假设参考已与周围环境物理兼容；当演示者、操作者与机器人 **不在同一环境** 时该假设失效。

### 方法总览：Perceptive Motion Tracking（PMT）

1. **TCRS（Terrain-Conformal Reference Synthesis）** — 离线将原始人体片段 + 采样高程场转为地形一致参考：接触感知落脚构造、足几何摆动优化（mid-foot 帧 MPPI）、支撑感知根重建、碰撞修复、多点 IK。**仅训练监督，部署不查询。**
2. **盲 adapted-reference teacher** — 无视觉 Transformer actor–critic，PPO 跟踪 TCRS 参考，暴露特权状态。
3. **Identity-gated vision student** — 部署策略接收 **原始参考 + 机器人中心高程扫描**；地形经 **tanh(α) 初始化≈0 的残差通路** 注入，初始等同纯跟踪器，仅在需要时学局部修正。
4. **Target-frame action alignment** — teacher 在 adapted 帧；student 围绕 raw 参考；蒸馏目标 `a* = (q_reftcrs + μ_tea) − q_refraw`，使跨参考帧蒸馏有意义。

### 架构（页面归纳）

- **Transformer motion-tracking backbone** 产出 motion intent `u_t`；**Map CNN → query-conditioned MapTransformer** 产出 terrain latent `z_vis`。
- Identity-gated fusion：`u′ = u + tanh(α_u)·Δu`；动作均值残差 `μ = μ_base + tanh(α_a)·r`。
- 目标：PPO + value + entropy；Huber 辅助损失（速度、anchor、足轨迹）。

### 量化（训练消融，页面数字）

| 对比 | 结果 |
|------|------|
| Full PMT vs 无视觉 | **54.6 vs 3.6**（均值 reward，末 1k/10k iter） |
| Transformer vs MLP/GRU/CNN 骨干 | **+5–8** reward |
| 有 vs 无 target-frame distillation | **54.6 vs 50.1**（+4.5） |
| TCRS vs Z-offset 基线（足-地形穿透） | **5.48 → 2.38 cm**（−56.6%）；间隙违规 −48.3% |

- 共享任务、奖励、观测契约与 **48×A800** 预算；去视觉伤害最大，说明增益来自 **感知与架构** 而非单纯容量。

### 真机与演示

- **Unitree G1** 单策略：楼梯、坡、稀疏块、凹障、草地、室内外不规则面；含 mocap 遥操作、杂技（后空翻上台阶等）、风格舞蹈、侧向/后退步态、户外人行道与广场。
- **浏览器 MuJoCo WASM + ONNX** 在线 demo（无需安装）。

### 局限（页面自述）

- TCRS 为 **运动学合成器**，不建模接触丰富动力学；假设 **静态刚体、可观测高程场**——不覆盖可变形、颗粒或湿滑介质。
- 适应以 **足部为中心**，上身命令原样保留 → **上肢可能与近障碰撞**（页面展示 mocap 撞障失败例）。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-perceptive-bfm.md`](../../wiki/entities/paper-perceptive-bfm.md)
- 任务挂接：[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)
- 概念互链：[`wiki/concepts/behavior-foundation-model.md`](../../wiki/concepts/behavior-foundation-model.md)、[`wiki/concepts/whole-body-tracking-pipeline.md`](../../wiki/concepts/whole-body-tracking-pipeline.md)
