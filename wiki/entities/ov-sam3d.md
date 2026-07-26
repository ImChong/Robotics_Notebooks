---
type: entity
tags: [repo, semantic-mapping, open-vocabulary, sam, point-cloud, offline, zju, tencent]
status: complete
updated: 2026-07-26
related:
  - ../queries/go2-3d-semantic-mapping-sam-pipeline.md
  - ./dualmap.md
  - ./ovo-semantic-mapping.md
  - ./point-lio-unilidar.md
sources:
  - ../../sources/repos/ov-sam3d.md
summary: "OV-SAM3D 是训练无关的开放词汇 3D 场景理解框架：超点粗 mask 经多视角 SAM 反投影修正，再结合 RAM 开放标签与重叠分数合并实例；偏离线点云理解。"
---

# OV-SAM3D

**OV-SAM3D**（[HanchenTai/OV-SAM3D](https://github.com/HanchenTai/OV-SAM3D)）是 **无需针对场景训练** 的开放词汇三维场景理解框架。

## 一句话定义

先从超点得到粗 3D mask，用多视角 **SAM** 反投影修正，再结合 **RAM** 开放标签与重叠分数合并实例——适合 **离线** 点云理解与伪标注，而不是 GO2 机载实时主路径。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SAM | Segment Anything Model | 多视角二维 mask 来源 |
| RAM | Recognize Anything Model | 开放世界图像标签 |
| OV | Open-Vocabulary | 不依赖闭集 3D 训练集的标签空间 |
| 3D | Three-Dimensional | 输出为三维实例 mask / 特征 |
| ScanNet | ScanNet Dataset | 常用室内评测集之一 |

## 为什么重要

- 把「多视角 SAM → 修正 3D 实例 → 开放标签」做成可跑通的离线流水线。
- 适合 GO2 语义落地的 **第二步**：几何锐利后，对保存的多视角图像 + 点云做离线着色/实例检查。
- 与 [DualMap](./dualmap.md) / [OVO](./ovo-semantic-mapping.md) 的在线路径形成互补。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 浙江大学（Zhejiang University）；腾讯优图（Tencent Youtu Lab） |
| 代码 | <https://github.com/HanchenTai/OV-SAM3D> |
| 开源 | **已开源** |
| 论文 | arXiv:2405.15580 |
## 核心原理

| 阶段 | 脚本/机制（对齐上游 README） |
|------|------------------------------|
| 粗 mask | `generate_coarse_masks.py`：superpoints + SAM |
| 细化 | `refine_masks.py`：开放标签 + 重叠分数 |
| 评测 | 对齐 OpenMask3D 风格评测脚本 |

## 工程实践

1. 准备 ScanNet 类扫描数据路径与 SAM checkpoint。
2. 先跑粗 mask，再 refine；确认输出 mask / feature 目录。
3. 用于 GO2：先用 Point-LIO 录包导出同步 PCD + 图像 + 位姿，再离线跑本流水线验证投影质量。

## 局限与风险

- **非实时 / 非 ROS 导航产品**：机载关键帧请选 DualMap 或自建轻量投影。
- 依赖多视角图像质量与标定；单雷达无相机无法单独发挥 SAM 优势。
- 开放标签噪声需后处理；勿直接写死进静态占据地图。

## 关联页面

- [GO2 三维语义建图与 SAM 流水线](../queries/go2-3d-semantic-mapping-sam-pipeline.md)
- [DualMap](./dualmap.md)
- [OVO](./ovo-semantic-mapping.md)
- [point_lio_unilidar](./point-lio-unilidar.md)

## 参考来源

- [sources/repos/ov-sam3d.md](../../sources/repos/ov-sam3d.md)
- 项目页：<https://hithqd.github.io/projects/OV-SAM3D/>
- arXiv：<https://arxiv.org/abs/2405.15580>

## 推荐继续阅读

- 上游仓：<https://github.com/HanchenTai/OV-SAM3D>
