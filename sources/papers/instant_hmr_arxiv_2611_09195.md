# InstantHMR: Real-Time Human Mesh Recovery on Your Phone（论文通读归档）

> 研究论文原文；2026-11-11 按整篇正文的引言、相关工作、方法、训练、评估协议、实验、消融、局限与结论整理，不以摘要替代。

- **标题：** InstantHMR: Real-Time Human Mesh Recovery on Your Phone
- **类型：** paper
- **作者：** Mohamed Adjel（NaturalPad）、Kahina Chalabi、Vincent Bonnet（LAAS-CNRS）
- **论文：** <https://arxiv.org/abs/2611.09195>；PDF <https://arxiv.org/pdf/2611.09195>
- **发表：** 2026-11-06，arXiv:2611.09195v1
- **代码：** <https://github.com/Naturalpad-studio/InstantHMR>
- **通读：** 已完成
- **通读范围：** 全文 1 Introduction；2 Related Work；3 InstantHMR（3.1 Network、3.2 Training）；4 Evaluation Protocol；5 Experiments；6 Limitations and Conclusion；图 1–3、表 1–4、参考文献及作者披露。补充数学细节在论文所指 supplementary 中，**不声称额外通读尚未取得的补充文件**。
- **关键结论：** 手机 NPU 优先的 MHR 参数回归路线；S23 NPU 5.49 ms/人、EMDB-1 48.8 mm PA-MPJPE；须对齐人体框、相机内参、MHR→SMPL 转换才能公平比较。

## 从问题到方法（§1–3§）

- **任务与难点（§1）**：既有 HMR 在桌面 GPU 上精度较高，但很难完整编译到手机 NPU；适合手机的方法精度通常不足。作者以手机 **30 FPS / 33 ms** 为实时阈值，避免仅看桌面 FPS。
- **表示与对比（§2）**：SMPL/SMPL-X/MHR 在顶点拓扑、骨架、单位方面不同；因而不能把各论文原始 PA-MPJPE 直接并列。MHR 将骨架与体表形状解耦。
- **网络（§3.1，图 2）**：288×288 人体 crop + CLIFF 条件 → HGNetV2 backbone（9×9 image tokens，512 通道）→ 4 层 8 heads Transformer、**71 queries** → 204 维 rig 参数 + 45 维形状、3 维相机平移、70×2×96 SimCC 关键点分布。3D joints 和 mesh 来自 MHR FK/蒙皮而非独立 3D head。
- **训练（§3.2）**：Meta SAM 3D Body 发布的人工标注 MHR 拟合；SA-1B（43%）、AI Challenger（26%）、Harmony4D（14%）、COCO（13%）、MPII（4%），共 4.2M crop；不含 3DPW/EMDB 测试图。11 项 loss 分参数、几何（含 595 顶点）及 2D+重投影监督。600 epochs × 400k samples，batch 256 / 4 GPU，AdamW + one-cycle peak lr 6e-4。
- **效率的解释（§5）**：NPU 擅长卷积/矩阵乘，NLF 的逐点 heatmap 查询带来大量 gather/reduce/transposes；InstantHMR 只回归一次 MHR 参数，再做确定性 FK/蒙皮，较适合 NPU。

## 评测协议（§4，表 1/2）

- 论文对 **11 个竞争方法**重新评分，使用尽量统一的 GT 框、真实焦距和 **Meta 官方 MHR→SMPL 转换**；3DPW 使用 H36M 14 joints，EMDB-1 使用 SMPL 24 joints。
- EMDB-1 每 10 帧取一帧（2,419 帧）、3DPW 每 10 帧取一帧（3,562 人帧），RICH 为 21,287 个测试 crop；没有预测的帧算缺失。
- 延迟按**单人、batch=1**、从 crop 到最多 6,890 顶点网格计时，排除 detector、crop 和视频解码。桌面使用 TensorRT，手机用 Qualcomm AI Hub 实机测量；未复现的对手结果如实描述，不认为原论文一定错误。

## 主要结果（§5，表 3/4）

| 模型 | 参数 | EMDB-1 PA-MPJPE | 3DPW PA-MPJPE | RICH PA-MPJPE | S23 NPU |
|---|---:|---:|---:|---:|---:|
| Nano / B0 | 20.0 M | 61.7 | 46.6 | 53.5 | 4.60 ms |
| Tiny / B1 | 20.3 M | 58.0 | 44.0 | 49.7 | 4.84 ms |
| Small / B2 | 24.4 M | 53.3 | 41.7 | 45.4 | 5.38 ms |
| Medium / B4 | 32.2 M | 48.8 | 40.0 | 41.7 | 5.49 ms |
| Medium-commercial / B4 | 32.2 M | 72.2 | 50.7 | — | 5.49 ms |

- PA-MPJPE 单位 **mm，越低越好**。Medium 在 RTX 4070 TensorRT 为 **0.98 ms**，S24 NPU **4.01 ms**，S23 NPU **5.49 ms**；论文称比更准确的 NLF-S / NLF-L 在 S23 NPU **快 12× / 23×**。
- **消融**：只用 standard crop 时 EMDB-1 **58.8 mm**，75% standard + 25% 其他三类 crop 后变 **48.8 mm**；对比 224 与 288 像素（早期配方）为 **54.2 → 49.8 mm**。不应把这个消融的 49.8 与最终模型的 48.8 混称。
- **精度并非绝对第一**：某些 GPU-only SAM 3D Body / Fast SAM 3D Body、更慢的 NLF 误差更小。端到端检测时序也不等同于论文模型延迟。

## 局限（§6）

1. 无专用手部编码器，手指只占全身 crop 中数个像素，手部细节受限。
2. 只在 Qualcomm Android NPU 评测，**未评测 iPhone Apple Neural Engine**。
3. 单图逐帧回归，论文未给出时间抖动/加速度指标；1 Euro 视频过滤属仓库功能，不能当作论文已验证的全链路动捕稳定性。
4. 训练图像的一部分仅许可研究或非商业用途，故 N/T/S/M 权重限制非商业；仅用 Harmony4D 与可商用 COCO 图像训练的 M-C 有约 **23 mm** EMDB-1 精度损失。

## 对 wiki 的映射

- [InstantHMR 主实体](../../wiki/entities/instanthmr.md)
- [源码通读归档](../repos/instanthmr.md)
