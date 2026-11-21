# GALATEA（arXiv:2609.10050）

> 来源归档（paper）；核对日期：2026-11-21。

- **标题：** GALATEA: Grounding Generated Video Plans in Simulation Towards Versatile Dexterous Controllers
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.10050>（v2，2026-09-12）
- **HTML 全文：** <https://arxiv.org/html/2609.10050v2>
- **PDF：** <https://arxiv.org/pdf/2609.10050>
- **项目页：** <https://boyuan-an.github.io/GALATEA/>
- **代码：** <https://github.com/boyuan-an/GALATEA>（2026-11-03 已公开 `recon/` 与 `rl/` 完整代码及 `rl-v1.0` 预训练权重/示例数据）
- **入库日期：** 2026-09-19
- **再次核对：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2609.10050v2` 全文（Sec. I–V，含 Seedance 2.0 生成、HOI 重建公式、Franka FR3 + Sharpa Wave 29-DoF RL 与 SAPG 训练、BC+DAgger 蒸馏、Table I–III 实验及失败分析），并复核 `boyuan-an/GALATEA@36b9048` 仓库最新开源状态；未在本地 GPU/真机运行。
- **通读：** 已完成
- **一句话说明：** 生成 HOI 视频（Seedance 2.0）→ MoGe-2 / SAM 2 / FoundationPose / WiLoR / HOI-DETR 联合优化重建 metric 轨迹 → Isaac Gym 仿真 SAPG 接触跟踪 → BC+DAgger 蒸馏跨物体灵巧策略；2500 生成片段约 2000 可用、1500+ 仿真落地，并在 FR3 + Sharpa Wave 真机 4 项未见任务取得 `27/40` 成功率。

## 开源状态

- **已开源（分模块混合许可，2026-11-03 已兑现发布）：** `boyuan-an/GALATEA`（commit `36b9048`）已完整发布 `recon/`（NVIDIA 非商业许可）与 `rl/`（GPL-3.0）代码，以及 `rl-v1.0` Release 下的 `power_drill`、`dog`、`old_camera2` 预训练权重与预处理/原始数据集（MANO 模型及部分第三方检测/跟踪权重需按原协议另行获取）。

## 核心摘录（arXiv:2609.10050v2 全文）

1. **视频生成与 HOI 重建（Sec. III-A）：** 基于真实首帧 RGB-D（提供 `D_1^st` 尺度锚点）与语言指令用 Seedance 2.0 生成三类操作视频（grasp-and-move、push-and-pull、pose-adjust）；用 MoGe-2 预测逐帧深度并在静态背景像素集上以截尾最小二乘对齐首帧深度；结合 SAM 2 掩码、FoundationPose 初始物体位姿、WiLoR 初始 MANO 手与 HOI-DETR 接触帧，联合优化轮廓、投影关键点、接触与时序平滑目标。
2. **仿真接地与蒸馏（Sec. III-B）：** 硬件为 Franka Research 3（7 DoF）+ Sharpa Wave（22 DoF）右手（共 29 维，30 Hz PD 控制）；每条参考做 5 组接触保持增强（手接近/接触后共变换）与 8× 减速 IK 预处理；使用非对称 Critic + SAPG 训练多轨迹专家（按物体类别分最多 2 个专家），再以 BC + DAgger 蒸馏为单一控制器。
3. **定量结果（Table I–III）：**
   - **重建：** 480 次人工盲评（120 段生成视频）87% 获第一（平均排名 `1.17` vs VideoManip `2.16`、DO AS I DO `2.88`、EgoInfinity `3.47`）；H2O / HO-Cap ADD-S 为 `4.82 cm` / `3.88 cm`；2500 生成片段中 `83%`（约 2000 条）通过质检，1500+ 在仿真成功落地。
   - **仿真跟踪：** 42 训练物体专家宏平均 `78.6%`（中位 `80.5%`）；蒸馏后仅降 `4.2` / `6.0` pp；未见基准轨迹 `66.6/68.8%`、5 个新物体 `54.2/52.2%`。
   - **真机闭环（Table III）：** 40 次未见视频计划（D455 + FoundationPose 在线跟踪）总体 `27/40`（`jar-neck` 6/10、`jar-top` 6/10、`mug-rim` 7/10、`mug-handle` 8/10）。
4. **作者自述局限（Sec. V）：** 未系统覆盖手内旋转（生成视频与重建噪声掩盖精细手指运动）；手掌摩擦非力封闭动作或贴桌薄物体抓取困难；接触切换处存在 Sim2Real 偏差，脱离参考轨迹后恢复有限。

## 对 wiki 的映射

- [paper-galatea](../../wiki/entities/paper-galatea.md)
- [generative-world-models](../../wiki/methods/generative-world-models.md)
- [manipulation](../../wiki/tasks/manipulation.md)
