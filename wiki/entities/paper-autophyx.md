---
type: entity
tags:
  - paper
  - robotics
status: complete
updated: 2026-10-28
project_id: autophyx
arxiv: "2610.29517"
project: https://hb-pencil-zero.github.io/AutoPhyX/
related:
  - ../methods/crisp-real2sim.md
  - ../comparisons/sim2real-vs-real2sim-fine-tuning.md
sources:
  - ../../sources/papers/autophyx_arxiv_2610_29517.md
  - ../../sources/sites/autophyx-project.md
summary: "以三维资产和文本为条件预测空间变化的杨氏模量、泊松比与密度体素场，供物理仿真使用。"
---

# AutoPhyX: Automatic Text-Condition Physics Property Generation

以三维资产和文本为条件预测空间变化的杨氏模量、泊松比与密度体素场，供物理仿真使用。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FEM | Finite Element Method | 有限元仿真
| MPM | Material Point Method | 材料点法
| VLM | Vision-Language Model | 数据构造中的语义推理 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 北京大学、北京通用人工智能研究院、安庆师范大学 |
| 论文 | [arXiv:2610.29517](https://arxiv.org/abs/2610.29517) |
| 项目页 | [官方网站](https://hb-pencil-zero.github.io/AutoPhyX/) |

## 方法与意义

以三维资产和文本为条件预测空间变化的杨氏模量、泊松比与密度体素场，供物理仿真使用。 系统用文本调制视觉体素特征，预测随空间变化的材料参数；训练数据由可控部件分解、材料参数与文字描述构成。网格、点云或高斯表示可通过体素场对接 FEM/MPM，但结果依赖输入文本和训练分布。

## 源码运行时序图

**不适用**：项目页无法访问，尚未核实可运行官方实现。

## 实验与评测

论文报告 Text2PhyX 上 PSNR 23.421→26.012，ABO-500 上平均位移误差相对次优方法降低 20%；均为作者实验结果。

## 结论

该工作提供了值得验证的管线，但实验指标应按原始设置解读。

- 先确认输入、输出及所需资产，再评估接入成本。
- 在目标动作、材料和仿真器上独立测量效果。
- 将仿真结果用于真机前，检查物理约束与安全边界。

## 工程实践与局限

项目页本次无法访问，未能核实代码、权重或数据的实际开放状态；论文中 Text2PhyX 含 1700 个资产及每个 8 组标注。参数预测不等于实物测量，应用前需校准。

## 关联页面

- [相关方法](../methods/crisp-real2sim.md)
- [相关概念](../comparisons/sim2real-vs-real2sim-fine-tuning.md)

## 参考来源

- [论文归档](../../sources/papers/autophyx_arxiv_2610_29517.md)
- [项目页归档](../../sources/sites/autophyx-project.md)


## 推荐继续阅读

- [原论文](https://arxiv.org/abs/2610.29517)
