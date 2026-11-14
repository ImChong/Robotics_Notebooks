---
type: source
title: "Sparsh: Self-supervised touch representations for vision-based tactile sensing — 原文精读归档"
url: "https://arxiv.org/abs/2410.24090"
date: 2026-11-14
status: ingested
tags: [paper, tactile-sensing, ssl, foundation-model, vbts, tacbench, meta, uw, cmu, corl]
---

# Sparsh — 原文精读归档

- **论文：** *Sparsh: Self-supervised touch representations for vision-based tactile sensing*
- **arXiv：** [2410.24090](https://arxiv.org/abs/2410.24090)（CoRL 2024）
- **作者与机构：** Carolina Higuera（Meta FAIR / UW）、Akash Sharma（Meta FAIR / CMU）、Chaithanya Krishna Bodduluri、Taosha Fan、Patrick Lancaster、Mrinal Kalakrishnan、Michael Kaess、Byron Boots、Mike Lambeta、Tingfan Wu、Mustafa Mukadam（Meta FAIR、University of Washington、Carnegie Mellon University）
- **项目页 / 代码：** <https://sparsh-ssl.github.io/> · <https://github.com/facebookresearch/sparsh> · <https://huggingface.co/collections/facebook/sparsh-67167ce57566196a4526c328>
- **核对范围：** 通读 `arXiv:2410.24090` 原文抽取文本（全文主文及附录 A–E.3，含第 1–9 节与 Appendix A 数据集策划、Appendix B 预训练超参、Appendix C TacBench [T1–T6] 任务设定与指标、Appendix D 重建监控 probe、Appendix E 消融实验表 Table 7–15，覆盖除参考文献末尾列表外的全部技术内容）。
- **通读：** 已完成

## 核心技术要点摘录

1. **统一的多传感预训练与预处理规范：** 汇总 4 个跨传感器视触觉数据集共约 **661k** 张无标注触觉图像（70% 即 **462.7k** 用于自监督预训练）：自研 Touch-Slide（DIGIT，9 物体 × 5 轨迹 = 180k）、YCB-Slide（DIGIT，180k）、Touch-and-Go（带标记点 GelSight 2017，220k）以及 ObjectFolder-Real（GelSight Mini，81k）。对无标记点传感器（DIGIT、GelSight Mini）做固定背景差分（background subtraction），对带物理标记点的 GelSight 2017 保留原始图像（因标记点携带切向变形信息）；将相隔 5 帧的帧对拼接为 6 通道输入 $I_t \oplus I_{t-5} \in \mathbb{R}^{h \times w \times 6}$（60 Hz 采样下约 80 ms 滑移响应时间窗，缩放至 $224 \times 224$），V-JEPA 则采 `[t, t-2, t-4, t-6]` 4 帧片段（约 100 ms）。
2. **四大 SSL 族系统对比（ViT-B/14）：** 在 8× NVIDIA A100 80GB 上统一用 AdamW 训练 150 epochs，并移除 `[cls]` token（DINO 复用 ViT register tokens）：
   - `Sparsh (MAE)`：86.25M 参数，RTX 3080 推理 **104 FPS**；
   - `Sparsh (DINO)`：86.26M 参数，**112 FPS**，EMA 0.998（另含 `DINOv2` 变体）；
   - `Sparsh (IJEPA)`：86.39M 参数，**112 FPS**，EMA 0.996（采用在线 DPT 重建探针在不回传梯度至主干的情况下监控隐空间收敛）；
   - `Sparsh (VJEPA)`：86.54M 参数，**60 FPS**（4 帧输入），EMA 0.996。
3. **TacBench [T1–T6] 基准与主实验结果：** 除 T1A（Dense Prediction Transformer 物理力场无监督反演）与 T6（Diffusion Policy）外，均采用 **冻结 ViT 主干 + Attentive Probe**（1 层 12 头 768 维交叉注意力 + 2 层 MLP）在 Full / 1/3 / 1/10 / 1/100 标签预算下评估：
   - **总体增益：** 在 33%–50% 标注预算下相对从零端到端训练（E2E）平均提升 **+95.1%**（摘要/正文均值；Appendix Table 13 按 T1-DIGIT +28.31%、T1-GelSight +59.74%、T2 滑移 +242.70%、T3 位姿 +235.89%、T4 抓取 +5.14%、T6 Bead Maze +19.72% 计算均值为 **+98.75%**）；其中 DINO 总体领先 IJEPA **+5.6%**（Table 13 细项均差 **+8.91%**）。
   - **T1 三轴力估计（25k 测试集 RMSE，mN）：** DIGIT 1/3 数据下 E2E `61.42` vs DINO `44.03`、**DINOv2 `26.85`**（1/100 数据 E2E `187.51` vs **DINO `97.95`**）；GelSight Mini 全量/1/3 数据下 E2E `57.21` / `59.09` vs **DINO `20.25` / `23.79`**、MAE `22.72` / `23.28`（1/100 数据 DINO `53.43` vs E2E `82.42`）。
   - **T1A 力场可视化：** 在 `[2, 5, 8, 11]` 层接无监督 DPT，结合光流重投影 MSE（法向）与 Charbonnier + SSIM + 平滑正则（剪切）输出像素级法向与剪切力分布图。
   - **T2 滑移检测（125k 样本，正样本 13%，F1）：** 1/3 数据下 E2E `0.238` vs **VJEPA `0.828`**、MAE `0.818`、IJEPA `0.791`；1/100 数据下 E2E `0.214` vs **VJEPA `0.760`**。
   - **T3 局部物体相对 $SE(2)$ 位姿（Allegro 指尖 DIGIT，每自由度 11 个对数均匀分箱，630 测试集准确率）：** Full / 1/3 数据下 E2E `0.812` / `0.245` vs **DINO `0.913` / `0.834`**、VJEPA `0.856` / `0.648`、IJEPA `0.851` / `0.601`。
   - **T4 抓取稳定性（Feeling of Success，GelSight'17，1.3k 测试集准确率）：** 1/3 数据 E2E `0.725` vs **IJEPA `0.782`**、DINOv2 `0.770`；1/10 数据 **DINO `0.773`**、IJEPA `0.768` vs E2E `0.682`。
   - **T5 织物 20 分类（GelSight'17，26k 测试集，随机基线 0.05）：** Full / 1/3 / 1/10 下 E2E `0.437` / `0.365` / `0.373` vs **MAE `0.599` / `0.588` / `0.527`**、VJEPA `0.580` / `0.545` / `0.507`；跨传感器 GelSight→DIGIT 10-shot 迁移（Table 14）中 DINO 达 **61.8%** vs E2E **10.9%**（zero-shot 9.1% vs 3.6%）。
   - **T6 Bead Maze 盲操（Franka + DIGIT，50 条遥操作约 34k 对，Diffusion Policy obs=2, pred=8, chunk=3cm/15 步）：** 验证轨迹误差（Table 12，Full / 1/10）E2E `8.46` / `9.80 mm` vs **IJEPA `5.47` / `5.46 mm`**、DINO `5.54` / `5.71 mm`；真机 10 组未见起点平均穿行距离（Table 11）E2E `6.70 cm`，**MAE 微调 `11.25 cm`**、DINO 冻结 `10.80 cm`、IJEPA 微调 `10.02 cm`（均相对 E2E 提升约 20%–53%，但因缺乏显式力控闭环，未见迷宫均未完全走通；单迷宫过拟合场景下 E2E 反而可达 ~80% vs Sparsh ~30%）。
4. **消融结论（Appendix E）：** 全量微调主要提升隐空间方法（IJEPA/VJEPA）在 T1/T3 连续回归任务上的精度，部分微调与冻结接近；模型缩放至 ViT-Small（384 维 vs Base 768 维）在 1/3 数据下导致 T1 DIGIT RMSE 恶化 **+74%**、GelSight 恶化 **+50.3%**、T3 掉点约 **20%**，而语义分类（T2/T4/T5）除 1% 极低样本外降幅有限。
