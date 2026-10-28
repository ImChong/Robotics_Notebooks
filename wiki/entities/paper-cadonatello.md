---
type: entity
tags: [paper, robotics, research]
status: complete
updated: 2026-10-28
project_id: cadonatello
arxiv: "2610.33575"
summary: "多部件参数化 CAD 重建：先对真实三维扫描分割部件，再用截面驱动草图参数化和拉伸重建各部件。"
related:
  - ./zero-to-cad.md
  - ../concepts/robot-simulation-three-layers.md
sources:
  - ../../sources/papers/cadonatello_arxiv_2610_33575.md
---

# CADonatello: Towards Real-World Multi-Part 3D CAD Reconstruction

**一句话定义：** 先对真实三维扫描分割部件，再用截面驱动草图参数化和拉伸重建各部件。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAD | Computer-Aided Design | 计算机辅助设计 |
| 3D | Three-Dimensional | 三维 |
| CC3D | CC3D Cross-Sections | 论文提出的截面配对数据集 |

## 方法

先对真实三维扫描分割部件，再用截面驱动草图参数化和拉伸重建各部件。

## 评测与结果

论文引入 CC3D Cross-Sections，将真实扫描截面图与配准 CAD 的参数化草图配对，以适应扫描噪声。

## 结论

这项研究提供了多部件参数化 CAD 重建的一个具体方案。

- 复现时先对齐论文任务和评测条件。
- 将方法收益与实际应用条件分开验证。
- 对代码及数据的可用性持续跟踪。

## 源码运行时序图

不适用：尚未核实可运行的官方实现。

## 局限与开放状态

结果依赖部件分割与截面、拉伸表示；未核实到官方代码或数据下载入口。

## 关联页面

- [相关方法](./zero-to-cad.md)
- [机器人仿真三层结构](../concepts/robot-simulation-three-layers.md)

## 参考来源

- [论文归档](../../sources/papers/cadonatello_arxiv_2610_33575.md)

## 推荐继续阅读

- [论文原文](https://arxiv.org/abs/2610.33575)
