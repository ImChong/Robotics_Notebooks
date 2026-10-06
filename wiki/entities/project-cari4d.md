---
type: entity
tags: [project, nvidia, human-object-interaction, 4d-reconstruction, computer-vision]
status: complete
updated: 2026-10-06
summary: "CARI4D 官方论文实现与 CoCoNet Native MHR 模型发布：单目人–物 4D 重建；研究代码与商用模型的交付和许可不同。"
related:
  - ./paper-cari4d.md
  - ./dataset-form-hoi.md
sources:
  - ../../sources/repos/nvlabs-cari4d.md
  - ../../sources/sites/cari4d-project-page.md
---

# CARI4D 项目与代码

该项目实现 [CVPR 2026 CARI4D 论文](./paper-cari4d.md)，对单目第三人称 RGB 视频中的人体与物体进行类别无关、米制尺度的 4D 重建。官方 [NVlabs/CARI4D](https://github.com/NVlabs/CARI4D) 仓库提供推理 demo、BEHAVE 示例、训练代码与检查点入口。

## 研究实现

- **输入：** 视频、人体/物体分割、人体与物体初始化状态及带纹理对象网格。
- **处理：** 基础模型初始化 → CoCoNet render-and-compare refinement → 接触一致性修正。
- **公开进度：** 仓库记录代码于 2026-02-28 发布，训练代码于 2026-04-04 发布；需按 README 准备容器、权重和依赖。
- **许可：** 研究代码以仓库当前 LICENSE 和依赖条款为准；它与后续 NVIDIA 商用模型卡是不同交付。

\`\`\`mermaid
flowchart TB
  repo["NVlabs/CARI4D 研究仓库"] --> setup["Docker 或实验性 Conda 环境"]
  setup --> inputs["视频、掩码、人体初始化、对象网格"]
  inputs --> inference["CARI4D 推理与接触细化"]
  inference --> outputs["人体姿态、对象轨迹、渲染结果"]
  repo --> training["训练代码与数据准备文档"]
\`\`\`

## CoCoNet Native MHR 模型发布

NVIDIA 后续发布了 [CARI4D CoCoNet Native MHR 模型卡](https://huggingface.co/nvidia/cari4d_commercial)。模型针对 96 帧视频片段，采用双分支 RGB 与 XYZ/掩码编码和时空特征融合，输出每帧 MHR 参数、刚体对象姿态及左右手 contact logits。模型卡报告 231M 参数，并注明在 NVIDIA A100 上验证。

- 模型卡标注可用于商业或非商业计算机视觉场景，但下载受 **NVIDIA Open Model Agreement** 管理；访问文件前需接受联系信息共享条款。
- 模型用途包括视觉估计、分析、可视化、数据整理与机器人感知，明确**不用于直接自主控制或生命攸关决策**。
- contact logit 是接触估计，不是力测量或物理可行性保证。

## 与 FORM-HOI 的关系

[FORM-HOI](./dataset-form-hoi.md) 是多视角人–物数据集，提供标定、视频、人体姿态、对象轨迹和度量网格。数据卡将 CARI4D 列为 HOI 重建训练用途。CARI4D 论文以单目 RGB 为目标输入；FORM-HOI 则提供多相机重建标注，任务和观测条件不同。

## 官方入口

- [NVlabs/CARI4D GitHub](https://github.com/NVlabs/CARI4D)
- [CARI4D 项目页](https://nvlabs.github.io/CARI4D/)
- [NVIDIA CARI4D CoCoNet Native MHR](https://huggingface.co/nvidia/cari4d_commercial)
- [FORM-HOI 数据集](./dataset-form-hoi.md)
