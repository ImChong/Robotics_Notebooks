# Hand Visibility Detector（arXiv:2608.11574）

> 来源归档（ingest）

- **标题：** Hand Visibility Detector: Per-Keypoint Visibility Estimation for Hands
- **短名：** Hand Visibility Detector / HVD
- **类型：** paper / hand-pose / visibility / keypoint / perception / annotation
- **arXiv：** <https://arxiv.org/abs/2608.11574>
- **PDF：** <https://arxiv.org/pdf/2608.11574>
- **HTML：** <https://arxiv.org/html/2608.11574>
- **代码：** <https://github.com/ryhara/hand_visibility_detector> — [`sources/repos/hand_visibility_detector.md`](../repos/hand_visibility_detector.md)
- **权重 / Demo：** <https://huggingface.co/ryhara/hand-visibility-detector> · <https://huggingface.co/spaces/ryhara/hand-visibility-detector>
- **作者：** Ryosei Hara（庆应 / AIST）、Masashi Hatano（东京大学）、Rintaro Yanagi（AIST）、Atsushi Hashimoto（欧姆龙 SINIC X）、Takuma Yagi（AIST）、Mariko Isogawa（庆应 / AIST）
- **机构：** 庆应义塾大学（Keio University）；产业技术综合研究所（AIST）；欧姆龙 SINIC X（OMRON SINIC X）；东京大学（The University of Tokyo）
- **版本：** arXiv:2608.11574（2026-08-12）
- **入库日期：** 2026-08-15
- **一句话说明：** 把 **逐关节手部可见性** 从 HPE 的辅助信号拆成独立任务：冻结 HaMeR / WiLoR 骨干，只训 0.83M 的 RTMPose 风格 visibility head；HInt mAP **0.931**，多视三角化重投影误差最多降 **10.1%**。官方仓 **已开源、可运行**（研究/非商用，继承上游许可）。

## 摘要级要点

- **问题：** 现成 HPE（HaMeR、WiLoR 等）几乎总输出关节坐标，不显式说「这个关节在图里能不能直接看见」。遮挡/出画时位置是推断出来的，下游三角化、标注、遥操作却常把所有点当同等可信。先前工作把 visibility 当姿态精度的辅助项，本身很少被系统评测。
- **方法：** 手裁剪图 \(I\) → 冻结预训练 HPE 的 ViT 骨干出特征图 \(F\) → 轻量 visibility head（1×1 压缩 + GAU + 空间均值 + sigmoid）出 21 维 MANO 关节可见概率 \(\hat{v}_j\in[0,1]\)。可见 = 未被遮挡且未出画。只训 head，BCE。
- **数据：** HInt（Pavlakos et al. 2024）人工逐关节可见标签；训练 25,273 / 评测 5,374。覆盖 web 与 egocentric。
- **主结果：** HInt mAP **0.931** / F1 **0.896**，相对 Kim et al. 2021 与 Contact4D 的 ImageNet CNN 头约 **+3.4 mAP**。骨干消融：HaMeR / WiLoR 最好；DINOv3 0.897；微调 WiLoR 骨干反而掉到 **0.622**。
- **下游：** 用 WiLoR 的 2D 关键点做 DLT 三角化；按本文逐关节可见性加权，相对无加权与「整手检测置信度加权」，DexYCB / HO3D / H2O 的重投影误差都下降，HO3D 均值最多 **−10.1%**。
- **开源（截至 2026-08-15）：** 无独立 `*.github.io` 项目页；以用户给出的 GitHub 与 HF 为准。`demo.py` / `demo_video.py` / `demo_gradio.py` + `training.train` / `evaluate` + HF `best.pt` / `best_hamer.pt` → **已开源、可运行**。许可为研究/非商用，叠加 WiLoR / HaMeR / MANO 等上游条款。

## 核心摘录（面向 wiki 编译）

### 架构

1. **输入：** GT 框（训/评）或 WiLoR 检测框（下游）；框扩 1.25×，resize 256×256，中心裁 256×192。
2. **Hand Encoder：** 冻结 HaMeR 或 WiLoR 的 ViT；16×16 patch → \(h=16,w=12,C=1280\)。
3. **Visibility Head：** 1×1 把 \(C\to d=256\) → flatten → FC → GAU（全局空间依赖）→ 1×1 到 \(J=21\) → 空间均值 → sigmoid。0.83M 参数，约占 631M 全模型的 **0.131%**。
4. **损失：** 逐关节 BCE；出画关节标不可见。
5. **训练：** 只训 head；AdamW \(10^{-3}\)；100 epoch；batch 256；单卡 H200 约 2.5 h / 10 GB。

### 数字读法

| 设定 | HVD | 对照 |
|------|-----|------|
| HInt mAP / F1 | **0.931 / 0.896** | Kim 0.895 / 0.858；Contact4D 0.897 / 0.860 |
| 骨干（同 head） | HaMeR 0.932；WiLoR 0.931 | DINOv3 0.897；ViT-H 0.838；CSPNeXt-X 0.800；ResNet-152 0.796 |
| 微调 WiLoR 骨干 | 0.622 | 冻结 0.931 — 任务微调会毁掉手结构先验 |
| Head 消融（冻 WiLoR） | full 0.931 | 无 GAU 0.887；Kim 线性头 0.905 |
| 阈值 0.3–0.7 | F1 > 0.88 | 峰值约 0.5 |
| 三角化重投影 | 三数据集中位/均值/IQR 都降 | HO3D 均值最多 **−10.1%**（视角少、物体遮挡重） |

### 开源核查（步骤 2.5）

无独立项目页。论文 Code 链与用户给出的 GitHub 一致。仓库核查见 [`sources/repos/hand_visibility_detector.md`](../repos/hand_visibility_detector.md)：推理包 + 训练/评测脚本 + HF 权重 + Gradio Space → **已开源、可运行**。许可 **非 OSI 宽松许可**（研究/非商用 + 上游叠加）。

## 对 wiki 的映射

- 升格 [Hand Visibility Detector 论文实体](../../wiki/entities/paper-hand-visibility-detector.md)
- 交叉：[WiLoR](../../wiki/methods/wilor.md)、[灵巧操作数据管线](../../wiki/queries/dexterous-manipulation-data-pipeline.md)、[灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md)、[MediaPipe](../../wiki/entities/mediapipe.md)、[自动化标注流水线](../../wiki/methods/auto-labeling-pipelines.md)

## 当前提炼状态

- [x] 方法 + HInt / 三角化数字 + 开源入口
- [x] wiki 实体、时序图与交叉引用
- [x] `sources/repos/`
