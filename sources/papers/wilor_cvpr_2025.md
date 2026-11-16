# WiLoR：野外 3D 手部定位与重建（CVPR 2025）

- **英文题目：** WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild
- **作者：** Rolandos Alexandros Potamias、Jinglei Zhang、Jiankang Deng、Stefanos Zafeiriou
- **出处：** CVPR 2025，12242–12254
- **CVF 官方论文：** https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html
- **PDF：** https://openaccess.thecvf.com/content/CVPR2025/papers/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.pdf
- **官方项目：** https://rolpotamias.github.io/WiLoR/
- **补充材料：** https://openaccess.thecvf.com/content/CVPR2025/supplemental/Potamias_WiLoR_End-to-end_3D_CVPR_2025_supplemental.pdf
- **官方代码：** https://github.com/rolpotamias/WiLoR
- **原有代码归档：** [sources/repos/wilor.md](../repos/wilor.md)
- **归档日期：** 2026-11-12
- **核对范围：** CVPR 2025 论文正文（第 1–6 节：引言、相关工作、WHIM 数据集、双阶段方法、检测/重建/动态实验、结论，及文末参考文献）；CVF Supplementary Material 第 1–5 节（训练实现、数据集对照、失败案例、14 个训练数据集和时间一致性）；官方项目页、GitHub README 与 demo.py 推理入口。
- **通读：** 已完成

## 关键摘录

- 由实时全卷积手部定位网络与 Transformer 3D 手部重建模块组成，覆盖多手、左右手区分与高保真对齐。
- 论文报告构造超过 200 万张野外手部图像，并无需额外时序模块就可将单帧估计应用到视频；这不等价于保证全局物理接触或不漂移。
- 对 4D-HOF：WiLoR 提供逐帧 MANO 状态与 2D 关节点，后续仍需 MoGe-3/相机参数、物体配准及条件桥联合细化。

## 完整方法拆解（CVPR §3–4）

1. **WHIM 数据建设：** 从 1,400 多个 YouTube 视频收集复杂光照、运动模糊、多人交互场景；先用 ViTPose/AlphaPose 找人体，再融合 MediaPipe、OpenPose 与 ContactHands 预测的手框；2D 手关节重投影 + 骨长/关节角生物力学约束 + ARCTIC 数据学习的 PCA prior 拟合 3D 手。检测训练集规模 >2M 图像。
2. **单阶段手部检测：** DarkNet 骨干输出 C3/C4/C5，多尺度 PANet neck 与 3 个 anchor-free heads 联合预测边界框、左右手类别和 2D keypoints；训练损失由 BCE、DFL、CIoU、keypoint L2 组成。**检测本身不输出 3D MANO**。
3. **Transformer 手部重建：** 单手 crop 进入预训练 ViT-Large，并添加 pose、shape、camera 三类 learnable tokens，初步预测 MANO pose/shape 和 weak-perspective camera。
4. **粗到精 refinement：** 将 ViT image tokens 还原成 2D feature map，再通过反卷积做多尺度上采样；把粗 MANO mesh 的顶点投影到各尺度特征图，采样并聚合 vertex-aligned features，回归姿态、形状、相机残差。损失包含 3D mesh、2D keypoints、MANO 参数和 adversarial pose prior。
5. **视频推理：** 逐帧运行 detector + reconstructor，不使用专门的时间网络；论文展示更小的 jitter，但不提供世界系/物理接触一致性的保证。

## 实验表与消融（原文的评测口径）

| 测试 | WiLoR | 对照 | 读法 |
|---|---:|---:|---|
| 检测 Proposed-M（RTX 4090） | **138 FPS** | ContactHands 29 FPS | 仅检测器，不是全栈推理 |
| 检测 Proposed-S（RTX 4090） | **175 FPS** | — | 更轻但 WHIM mAP 46.50，低于 M 的 53.79 |
| OxfordHands mAP（M） | **48.98** | ContactHands 36.41 | 相同评估设置 |
| FreiHAND PA-MPJPE (mm) | **5.5** | HaMeR 6.0 | Procrustes 对齐后手关节误差 |
| FreiHAND PA-MPVPE (mm) | **5.1** | HaMeR 5.7 | 对齐后网格顶点误差 |
| HO3Dv2 PA-MPJPE (mm) | **7.5** | HaMeR 7.7 | 注意是 HO3D v2 |
| 动态 HO3D Jitter | **5.92** | HaMeR 20.43 | 本论文定义的时间稳定性指标 |
| FreiHAND 去 refinement PA-MPJPE | 6.1（完整 5.5） | — | 多尺度 refinement 对精度有贡献 |
| FreiHAND 单尺度 refinement PA-MPJPE | 6.0（完整 5.5） | — | 多尺度优于单尺度 |

原文 Table 1–6；不能将 detector 的 138/175 FPS 与完整手部 3D reconstruction 推理吞吐混同，也不能把 HO3Dv2 指标误引为 4D-HOF 的 HO3D-v3 指标。

## Supplementary 补充结论

- **检测训练：** 200 epoch，Adam，初始学习率 0.01，最后 30 epoch 线性到 1e-6；两张 RTX 4090、batch 256、三周（作者报告）；loss 权重 (0.5, 1.5, 15, 10)。
- **重建训练：** ViT-Large/ViTPose 初始化，隐藏维度 1280，16×12 特征图；pose 96 个 6D rotation 参数、shape 10 个、camera 3D translation；Adam，1000 epoch，初始 LR 1e-5，weight decay 1e-4；训练混合 14 个 2D/3D 数据集约 4.2M 图像，和 WHIM >2M 检测集**不是同一统计**。
- **失败案例：** 极端手指姿势、拥挤场景小手漏检、双手接触的 3D 交互关系不受直接约束；相机系预测不能自动提供 metric world-space 手轨迹。
- **开放边界：** 官方 README 提供 `python demo.py --img_folder demo_img --out_folder demo_out --save_mesh --fast`，需要 `detector.pt`、`wilor_final.ckpt` 和另行授权的 MANO_RIGHT.pkl。模型许可为 CC-BY-NC-ND，外部 MANO/Ultralytics 另有许可证；不应默认允许商用。
- **与 4D-HOF 关系：** 4D-HOF 将此处逐帧 MANO 初值与 MoGe-3 尺度、FoundationPose 物体姿态融合，再对手-物体时序做条件流匹配优化；单一 WiLoR 并未完成 HOI 接触重建。

## 对 wiki 的映射

- [复用现有 WiLoR 方法实体](../../wiki/methods/wilor.md)（不新增同名重复实体）
- [4D-HOF 论文实体](../../wiki/entities/paper-4d-hof.md)
