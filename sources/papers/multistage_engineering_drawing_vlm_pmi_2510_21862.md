# A Multi-Stage Hybrid Framework：多视图工程图纸的三阶段 PMI 自动解析

> 一手论文全文归档（paper / methods / inspection），与泊松比的二手综述分别保存。

- **英文标题：** A Multi-Stage Hybrid Framework for Automated Interpretation of Multi-View Engineering Drawings Using Vision Language Model
- **中文标题：** 多视图工程图纸自动解析：布局检测、旋转标注定位与双 VLM 混合框架
- **作者：** Muhammad Tayyab Khan, Zane Yong, Lequn Chen, Wenhe Feng, Nicholas Yew Jin Tan, Seung Ki Moon
- **机构：** A*STAR SIMTech、A*STAR ARTC、新加坡南洋理工大学
- **arXiv：** 2510.21862v2（2026-01-23）
- **论文 HTML 题录：** https://arxiv.org/abs/2510.21862
- **论文 PDF：** https://arxiv.org/pdf/2510.21862
- **代码：** 未从本论文题录、全文及作者官方公开入口核实到与此工作对应的可运行仓库；不等于断言从未开源。
- **入库日期：** 2026-11-19
- **核对范围：** 已逐节阅读 arXiv:2510.21862v2 **10 页完整 PDF**：摘要、引言、方法（§2）、数据集构建（§3）、全部实现及结果（§4）、讨论（§5）、结论/限制（§6）及论文参考文献；并逐图检查 Fig. 1–4 的三阶段流程/样例/数据分布/检测混淆矩阵与 Table 1 的每类精度、召回、F1、幻觉率。未独立复现实验或取得训练数据/权重。
- **通读：** 已完成

## 原文技术路线

1. **Stage 1（布局检测）**：YOLOv11-det 对整张工程图做 Views / Title Block / Notes 三类区域检测，训练集 1,000 张，划分 80%/20%。Table/混淆矩阵所示对角项分别约 0.96 / 0.99 / 0.98，这些为检测分类表现。
2. **Stage 2（定向标注）**：对每个 View 单独运行 YOLOv11-obb，识别 Measures（含半径/螺纹）、GD&T 和 Surface Roughness。训练集 1,406 张，标注量分别 9,663 / 3,215 / 152。混淆矩阵对角项 Measures 0.95、GD&T 0.97、Roughness **0.54**，低样本类别仍是明显瓶颈。
3. **Stage 3（专用解析）**：Title Block / Notes 的 **Alphabetical Donut VLM 采用 zero-shot，无微调**；数值标注的 **Numerical Donut VLM 经微调**并按 schema 输出 Unicode 工程符号。训练样本约 15,000 对图像/文本，文本约 1,585，对数字约 13,000。
4. **训练与输出**：数值模型按 70:20:10 划分 train/val/test，RTX 5090 上训练 30 epochs（AdamW、cosine LR、初始 1e-6、batch 4、FP16）。各分支 JSON 归并成统一图纸视图/标题栏/注释与标注结构；论文展示生成结果，但非未经人工复核即可投入产线的证明。

## Table 1（论文第 8 页，人工逐表核对）

| 分支 | 类别 | Precision | Recall | F1 | Hallucination |
|---|---|---:|---:|---:|---:|
| 文本（zero-shot） | Title Block | 0.522 | 0.545 | 0.533 | 0.478 |
| 文本（zero-shot） | Notes | 0.681 | 1.000 | 0.810 | 0.319 |
| 文本（zero-shot） | Overall | 0.601 | 0.773 | 0.672 | 0.399 |
| 数值（fine-tuned） | Measures | 0.864 | 0.991 | 0.923 | 0.136 |
| 数值（fine-tuned） | GD&T | 0.933 | 1.000 | 0.965 | 0.067 |
| 数值（fine-tuned） | Roughness | 1.000 | 1.000 | 1.000 | 0.000 |
| 数值（fine-tuned） | Overall | 0.932 | 0.997 | 0.963 | 0.067 |

**避免两个指标误读：**

- 数值与文本的 F1 存在明显差异，但**两个分支训练设置和样本量也不同**；不能仅归因为“模型天生更善于读数值”。
- Roughness VLM 的 **F1=1.0 是已定位到的 patch 在解析阶段的分数**；其上游 OBB 检测混淆矩阵对角项只有 **0.54**，所以端到端不可能直接按“粗糙度 100% 正确”来宣称。

## 局限与后续工作（§5–6）

- 论文明确认为**标题栏密集表格与自由文本识别偏弱**，文本 Donut 未针对该任务微调，后续应改进 text-centric parsing。
- 粗糙度仅有 152 处 OBB 标注，检测是长尾短板；需要平衡数据、合成增强。
- 未来计划高级 VLM 与生成结果交叉验证；这属于**未来工作**，不应视为已交付的生产级安全校验。
- 论文的三阶段评测不是包含人工审核、气球编号、CMM 规划及最终验收的完整端到端质量检验准确率。

## 对 wiki 的映射

- [本论文独立实体](../../wiki/entities/paper-multistage-engineering-drawing-vlm-pmi.md)
- [VLM 与传统 OCR 的图纸 PMI 可靠性比较](../../wiki/comparisons/pdf-engineering-drawing-pmi-ocr-vlm.md)

## 参考来源

- [arXiv:2510.21862v2 PDF 全文](https://arxiv.org/pdf/2510.21862)
- [arXiv 作者与版本记录](https://arxiv.org/abs/2510.21862)
