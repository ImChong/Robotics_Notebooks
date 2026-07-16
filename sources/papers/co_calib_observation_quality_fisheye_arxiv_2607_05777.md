# Observation Quality Matters: Robust Multi-Fisheye Calibration via Failure-Oriented Analysis（arXiv:2607.05777）

> 来源归档（ingest）

- **标题：** Observation Quality Matters: Robust Multi-Fisheye Calibration via Failure-Oriented Analysis
- **类型：** paper / multi-fisheye calibration / camera calibration / bundle adjustment / perception
- **arXiv abs：** <https://arxiv.org/abs/2607.05777>
- **PDF：** <https://arxiv.org/pdf/2607.05777>
- **代码（即将开源）：** <https://github.com/HKUST-Aerial-Robotics/CO-Calib>
- **发表日期：** 2026-07（arXiv）
- **机构：** 香港科技大学（HKUST）电子与计算机工程系
- **作者：** Peize Liu, Zhe Tong, Chen Feng（通讯作者）, Shaojie Shen
- **入库日期：** 2026-07-16
- **一句话说明：** 通过 **failure-oriented 分析** 揭示多鱼眼标定失败的主因是 **内参初始化病态**（焦距–投影形状参数耦合、径向覆盖不足），而非检测召回或图像平面分布失衡；提出 **plug-in** 框架 **CO-Calib**（学习型标定板检测 + 误差分析引导帧选择），在不改 BA 后端的前提下将合成基准成功率 **68.1%→99.3%**，并在 Hex-Fisheye 实机全成功。

## 摘要级要点

- **问题背景：** 移动机器人与数据采集平台广泛采用 **多鱼眼大 FoV** 相机；随相机数量、布局与 FoV 增大，联合 BA 更紧耦合，标定对 **观测质量** 极度敏感，工业流程常靠经验采集规则，鲁棒性差。
- **现有管线（Kalibr 类）：** 分阶段初始化/精化 **内参、每帧标定板位姿、相机间外参**；后续工作（TartanCalib、MC-Calib 等）从畸变建模或 richer target 增强约束，但 **观测如何影响优化可解性** 仍缺乏系统刻画。
- **Failure-oriented 分析（Sec. III）：** 16 组合成设置（4 FoV × 4 双目相对 yaw）；每配置 100 序列 × 480 帧。
  - **失败定位：** 宽 FoV（220°/240°）失败率陡升；失败 trial 中 **~98–100% 归因于内参初始化阶段**。
  - **假设 1 — 边缘召回不足：** 几何检测器在边界 recall 下降；但换 **全 GT 观测** 后成功率反而 **68.1%→53.7%** → **非主因**。
  - **假设 2 — 图像平面分布失衡：** 配置间分布可差异大，但 success/failure 的 GT 分布与随机划分几乎不可分（$\Delta_{\mathrm{sp}}\le 0.13$ pp）→ **非主因**。
  - **主因 — 内参初始化病态：** 对 pose-free Schur 块 $\tilde{\mathbf{S}}_c^{(k)}$ 的条件数 $\gamma_k=\log_{10}\kappa(\cdot)$，失败 trial 显著更高；**窄径向 span** 使焦距尺度与 fisheye 投影形状参数（$\xi,\alpha,\beta$ 等）Jacobian 方向局部相似 → **focal–projection coupling**。
- **CO-Calib 框架（Sec. IV）：** **不改** 相机模型与 BA 后端，只构造 **optimization-ready** 观测序列。
  1. **学习型标定板检测器：** 在线物理 grounded 数据生成；边缘畸变区 recall/定位优于几何法（240° FoV recall **0.926 vs 0.684**）。
  2. **Coverage- & observability-aware selector：**
     - **Projective isotropy** $s_{\mathrm{iso}}=\sigma_{\min}/\sigma_{\max}$（单应 Jacobian 奇异值比）— 稳定位姿初始化代理；
     - **Directed radial span** $s_{\mathrm{drs}}$ — 激励径向投影变化、解耦焦距与投影形状；
     - **三阶段选帧：** Anchor（初始化友好）→ Co-visible（多相机共视外参）→ Mono-fill（补覆盖）。
- **主要结果（合成，Table VI）：** vs Kalibr — SR **68.1%→99.3%**；外参误差 **0.54/0.029→0.18/0.021**（平移 mm / 旋转 deg）；Random-subset **30.9%**；去掉 initialization 的 BA-only **13.5%**。
- **实机（Table VII）：** 标准双目各 yaw **5/5** 成功；**Hex-Fisheye** Kalibr **0/10**、CO-Calib **10/10**；与 Basalt 比 extrinsic 一致性更稳。
- **工程含义：** 标定采集应优先保证 **宽径向覆盖的可分离内参观测**，而非单纯「多帧/均匀撒点」；CO-Calib 可作为 Kalibr 等管线的 **前置 plug-in**。

## 对 wiki 的映射

- 沉淀实体页：[paper-co-calib-multi-fisheye-calibration.md](../../wiki/entities/paper-co-calib-multi-fisheye-calibration.md)
- 交叉：[三维坐标变换（标定语境）](../../wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)、[AprilTag（标定板 fiducial）](../../wiki/entities/april-tag.md)、[VINS-Fusion（HKUST 空中机器人组）](../../wiki/entities/vins-fusion.md)、[OpenVINS（标定工具链）](../../wiki/entities/open-vins.md)、[Visual Servoing（相机标定依赖）](../../wiki/methods/visual-servoing.md)

## 参考来源（原始）

- arXiv:2607.05777
- 对比基线：Kalibr、Basalt、TartanCalib（arXiv）、MC-Calib
