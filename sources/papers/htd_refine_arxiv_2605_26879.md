# HTD-Refine: Natural Human Motion Recovery by Aligning High-Order Temporal Dynamics from Monocular Videos（arXiv:2605.26879）

> 来源归档（ingest）

- **标题：** Natural Human Motion Recovery by Aligning High-Order Temporal Dynamics from Monocular Videos
- **简称：** HTD-Refine
- **类型：** paper / human motion recovery / monocular video / post-processing / CVPR 2026
- **venue：** CVPR 2026（项目页标注 Oral Award Candidate）
- **原始链接：**
  - arXiv abs：<https://arxiv.org/abs/2605.26879>
  - arXiv PDF：<https://arxiv.org/pdf/2605.26879>
  - 项目页：<https://zju3dv.github.io/htd-refine/>
  - 代码：Coming Soon（截至 2026-06-04 项目页标注）
- **机构：** 浙江大学（State Key Lab of CAD&CG）；Ant Group；The University of Texas at Austin
- **作者：** Dingkun Wei*、Zehong Shen*、Yan Xia、Georgios Pavlakos、Yujun Shen、Xiaowei Zhou（* equal contribution）
- **入库日期：** 2026-06-04
- **一句话说明：** 针对单目 HMR 轨迹「位置准但动力学不自然」（过平滑或抖动）的问题，用 **PVA-Net** 从视频显式预测 **2D 关键点 + 相机系 3D 速度/加速度**，再以全局优化 **后处理** 任意现有 HMR 初始化（TRAM / GVHMR / Human3R 等），显著降低 jitter、脚滑并恢复高频运动细节。

## 摘要级要点

- **问题：** 单目 world-grounded HMR 即使关节位置误差低，也常 **过平滑** 或 **抖动**；人类运动对微小误差极敏感，30 FPS 训练/评测进一步欠拟合高频瞬态。
- **观察：** 一阶 **速度** 受单目尺度与相机漂移污染；二阶 **加速度** 对慢变漂移更鲁棒，更能刻画起停、转向等事件——应作为显式监督与优化目标。
- **HTD-Refine：** **通用后处理框架**，不替换基线 HMR，而是在其 world-space SMPL 初始化上，用 PVA-Net 预测的高阶动力学 + 2D 关键点做序列级能量最小化。
- **PVA-Net：** 冻结 **ViTPose-L** 提特征 + **8 层时序 Transformer（RoPE）** + 三头解码 **关键点 / 速度 / 加速度**；在 BEDLAM、RICH、H36M 上训练。
- **优化：** 对 pose $\theta$、根朝向 $\Gamma$、平移 $\tau$ 最小化 $E_V + E_A + E_K + E_{\text{jerk}} + E_{\text{reg}}$；后处理用速度阈值做 **接触静止概率** + IK 抑制脚滑。
- **实验：** RICH（静态相机）、EMDB-2（移动相机）；相对 TRAM / GVHMR / Human3R **+HTD-Refine** 一致改善 Jitter、MPJVE、MPJAE、WA/W-MPJPE；新增 **MPJVE / MPJAE** 动力学指标。

## 核心摘录（面向 wiki 编译）

### 1) 三阶段管线

- **链接：** 论文 §3、Fig. 2；[项目页 Methods](https://zju3dv.github.io/htd-refine/)
- **摘录要点：**
  1. **Initialization：** 现成 HMR（TRAM / GVHMR 等）+ 相机外参 → 相机系 SMPL → 式 (1)(2) 变换到 world 根位姿。
  2. **PVA-Net：** 输入整段视频，输出 stabilized 2D keypoints、相机系 $V_c^t$、$A_c^t$。
  3. **Motion optimization：** 对 SMPL 参数做 Adam 迭代，对齐 PVA 预测 + 2D 重投影 + jerk 平滑 + 贴近初始化正则；可选接触 IK 后处理。
- **对 wiki 的映射：**
  - [HTD-Refine](../../wiki/entities/paper-htd-refine-monocular-hmr.md) — Mermaid 主干流程

### 2) PVA-Net 设计与训练

- **链接：** 论文 §3.2、Fig. 3
- **摘录要点：**
  - 速度：有限差分 $V_c^t = (\mathbf{J}_c^t - \mathbf{J}_c^{t-1})/\Delta t$；加速度：二阶差分式 (4)。
  - 损失：$L_H + L_V + L_A + L_{\text{tgm}}$（heatmap MSE + 速度/加速度 MSE + 2D 时序梯度匹配，借鉴 VDA）。
  - ViT 冻结，仅训时序模块与三解码头；RoPE 强调 motion onset / reversal / rhythm。
- **对 wiki 的映射：**
  - [HTD-Refine](../../wiki/entities/paper-htd-refine-monocular-hmr.md) — PVA-Net 机制表

### 3) 与隐式平滑 / 生成先验的对比

- **链接：** 论文 §1–2 Related Work
- **摘录要点：**
  - TRAM / GVHMR / WHAM 等 **隐式时序正则** 或自回归 rollout 仍难恢复高频；额外滤波常压制真实运动。
  - HuMoR / RoHM / LEMO 等 **生成/平滑先验** 改善感知质量但易与 2D 证据漂移或算力高。
  - HTD-Refine **显式** 对齐视频预测的速度–加速度场，在保留 2D 约束下直接优化动态保真度。
- **对 wiki 的映射：**
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md) — 视频源后处理选项
  - [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md) — 参考采集质量链

### 4) 主结果（项目页 / Table 1–2）

- **链接：** [项目页 Main Results](https://zju3dv.github.io/htd-refine/)；论文 §4
- **摘录要点（EMDB-2，移动相机）：**
  - TRAM → +HTD-Refine：Jitter **25.1→6.6**，MPJAE **12.3→8.0**，WA-MPJPE **78.8→71.7**。
  - GVHMR → +HTD-Refine：Jitter **17.2→7.2**，WA-MPJPE **118.7→69.2**，W-MPJPE **292.7→192.4**。
  - Human3R → +HTD-Refine：Jitter **529.6→132.5**，MPJAE **143.3→39.4**。
- **RICH（静态相机）：** TRAM Jitter **18.7→4.2**；GVHMR **13.0→3.6**；位置与 RTE 同步改善或持平。
- **对 wiki 的映射：**
  - [HTD-Refine](../../wiki/entities/paper-htd-refine-monocular-hmr.md) — 实验与 baseline 对照

## 对 wiki 的映射（汇总）

- [paper-htd-refine-monocular-hmr.md](../../wiki/entities/paper-htd-refine-monocular-hmr.md) — 主沉淀页
- 交叉更新：[motion-retargeting-pipeline.md](../../wiki/concepts/motion-retargeting-pipeline.md)、[whole-body-tracking-pipeline.md](../../wiki/concepts/whole-body-tracking-pipeline.md)、[motion-retargeting-gmr.md](../../wiki/methods/motion-retargeting-gmr.md)

## 引用（项目页 BibTeX）

```bibtex
@inproceedings{wei2026htdrefine,
  title     = {Natural Human Motion Recovery by Aligning High-Order Temporal Dynamics from Monocular Videos},
  author    = {Wei, Dingkun and Shen, Zehong and Xia, Yan and Pavlakos, Georgios and Shen, Yujun and Zhou, Xiaowei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
