---
type: entity
tags: [paper, human-object-interaction, 4d-reconstruction, monocular-video, nvidia, cvpr]
status: complete
updated: 2026-10-06
arxiv: "2512.11988"
venue: "CVPR 2026"
summary: "从单目 RGB 重建类别无关、米制尺度的人–物 4D 交互；融合基础模型假设、render-and-compare 与接触物理细化。"
related:
  - ./project-cari4d.md
  - ./dataset-form-hoi.md
  - ./paper-hoi-retarget.md
sources:
  - ../../sources/papers/cari4d_arxiv_2512_11988.md
  - ../../sources/repos/nvlabs-cari4d.md
  - ../../sources/sites/cari4d-project-page.md
---

# CARI4D：类别无关的人–物交互 4D 重建

**论文：** [CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction](https://arxiv.org/abs/2512.11988)  
**会议：** CVPR 2026 camera-ready。作者来自 NVIDIA、University of Tübingen、Tübingen AI Center 与 Max Planck Institute for Informatics。

## 研究问题与方法

单目 RGB 视频缺少真实深度和已知物体模板，遮挡与接触又使逐帧人体/物体估计容易漂移。CARI4D 从普通单目视频恢复时间、空间一致且具有米制尺度的人体与刚体物体交互轨迹，并面向未见对象类别和野外视频泛化。

方法流程：

1. 组合人体、物体形状/姿态和深度等基础模型预测，生成姿态假设。
2. 通过 pose-hypothesis selection 筛选兼容假设，再以可学习的 render-and-compare 目标联合对齐空间、时间和像素。
3. 基于人–物接触关系继续细化姿态，使结果更符合物理约束。
4. 输出人体与对象的 4D 状态，供动作分析、数据整理或机器人学习使用。

\`\`\`mermaid
flowchart LR
  video["单目 RGB 视频"] --> hypotheses["基础模型姿态与形状假设"]
  hypotheses --> selection["假设筛选与联合对齐"]
  selection --> contact["接触一致性细化"]
  contact --> tracks["米制尺度的人体与物体 4D 轨迹"]
  tracks --> downstream["可视化、数据整理、机器人学习"]
\`\`\`

## 论文报告结果

| 项目 | 摘要报告 |
|---|---|
| 主要场景 | BEHAVE、InterCap 等人–物交互 |
| 重建误差 | 相比先前方法，分布内改善 38%，未见数据改善 36% |
| 泛化 | 可零样本处理训练类别之外的互联网视频 |

百分比指重建误差改善，不代表机器人任务成功率，也不应与 FORM-HOI 的数据规模混为一谈。

## 与相邻资源的关系

- [CARI4D 项目与实现](./project-cari4d.md)：原 CVPR 研究代码及后来发布的 CoCoNet Native MHR 商用模型卡分开说明。
- [FORM-HOI 数据集](./dataset-form-hoi.md)：多视角人–物序列与轨迹，可用于 HOI 重建训练和机器人策略 grounding。
- [HOI-Retarget](./paper-hoi-retarget.md)：把重建的人–物轨迹用于机器人动作重定向，属于下游流程。

## 局限与使用边界

单目重建会受遮挡、初始化、分割、深度和对象网格质量影响；近似对称物体也可能造成姿态歧义。接触预测是几何/运动线索，不等价于接触力、摩擦或物理可行性。下游机器人部署需独立验证轨迹与接触，不能将重建输出直接当作安全控制命令。

## 来源

- [arXiv 论文及 v3 版本](../../sources/papers/cari4d_arxiv_2512_11988.md)
- [官方代码仓库](../../sources/repos/nvlabs-cari4d.md)
- [项目页与模型发布](../../sources/sites/cari4d-project-page.md)
