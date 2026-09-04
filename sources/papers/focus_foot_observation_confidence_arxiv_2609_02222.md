# FOCUS: Foot Observation Confidence for Robust Humanoid Proprioceptive Odometry（arXiv:2609.02222）

> 来源归档（ingest）

- **标题：** FOCUS: Foot Observation Confidence for Robust Humanoid Proprioceptive Odometry
- **全称展开：** Foot Observation Confidence from Unannotated Simulation
- **类型：** paper / humanoid / odometry / ekf / contact-reliability
- **arXiv：** <https://arxiv.org/abs/2609.02222>
- **PDF：** <https://arxiv.org/pdf/2609.02222>
- **HTML：** <https://arxiv.org/html/2609.02222>
- **项目页：** 论文与检索 **未见** 独立 `*.github.io` 页
- **代码：** 截至 2026-09-04 **确认未开源**（无官方 GitHub / 权重；勿与 [StefanoFerraro/FOCUS](https://github.com/StefanoFerraro/FOCUS) 物体中心世界模型仓混淆）
- **机构：** 武汉大学（WHU）、智元机器人（AgiBot）
- **平台：** AgiBot A3 Ultra（174 cm / 60 kg；五台同构机）
- **入库日期：** 2026-09-04
- **最后更新：** 2026-09-04
- **一句话说明：** 用连续足部 FK 可靠度权重替代二值接触门控，在 EKF 里混合 FK 速度与 IMU 传播并调制观测噪声。

## 开源状态（步骤 2.5，2026-09-04）

| 组件 | 状态 |
|------|------|
| 项目页 | **未见** |
| GitHub / ONNX / 权重 | **未见** |
| 论文承诺 | 未列将开源 URL |

**结论：确认未开源**

## 核心论文摘录（MVP）

### 1) 接触 ≠ FK 可信

- 动态步态常见部分支撑、拖趾、打滑；二值接触会把整只脚的 FK 观测硬切进/出滤波器。
- FOCUS 输出 \(w_L,w_R\in[0,1]\)：饱和斜坡 \(\tau_i=\min(1,2.5 w_i)\) 混合 FK 与 IMU 速度，并把 \(R_{\mathrm{vel}},R_p,R_h,Q_f\) 按 \(1+(1-w)S\) 放大。
- **对 wiki 的映射：** [paper-focus-foot-observation-confidence](../../wiki/entities/paper-focus-foot-observation-confidence.md)、[EKF](../../wiki/formalizations/ekf.md)、[接触估计](../../wiki/concepts/contact-estimation.md)

### 2) 无人工连续标签：仿真 FK 加权速度一致性

- Isaac Lab 冻结运动跟踪策略回放 5420/1151 条；主损失 \(\mathcal{L}_{\mathrm{FK}}+\ 0.3\mathcal{L}_{\mathrm{BCE}}\)，部署输入仅 IMU + 12 维下肢关节（无力矩）。
- 因果 Transformer：50 帧 × 30 维 → 4 层 4 头 → ONNX 双 sigmoid；693k 参数，i7-13700K 单线程 1.62 ms。
- **对 wiki 的映射：** [paper-focus-foot-observation-confidence](../../wiki/entities/paper-focus-foot-observation-confidence.md)、[状态估计枢纽](../../wiki/overview/hub-state-estimation.md)

### 3) 真机走路 ATE −70.8%，动态套路 −42.7%

- 仿真走路：ATE 1.016 → 0.166 m（相对力矩阈值，−83.7%）。
- 19 段真机走路（五台机、1.51 km）：2.634 → 0.768 m。
- 四套真机舞蹈：0.947 → 0.542 m；消融显示学二值接触反而更差（6.125 m）。
- **对 wiki 的映射：** [paper-focus-foot-observation-confidence](../../wiki/entities/paper-focus-foot-observation-confidence.md)、[X-IONet](../../wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md)

## 当前提炼状态

- [x] 论文摘要与评测表摘录
- [x] 开源核查（无项目页、无官方仓）
- [x] wiki 映射
