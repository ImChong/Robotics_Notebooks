---
type: entity
tags: [connectomics, dataset, drosophila, janelia, hhmi-janelia, neuroscience, sexual-dimorphism]
status: complete
updated: 2026-09-10
related:
  - ../concepts/fly-connectomics-stack.md
  - ./flywire.md
  - ./neuprint.md
  - ./neuroglancer.md
  - ./dvid.md
  - ./flybrainlab.md
sources:
  - ../../sources/sites/male-cns-connectome.md
summary: "首个完整 proofread 的雄性果蝇 CNS（脑+视叶+VNC）突触分辨率连接组 v1.0，CC-BY 开放，可与雌性 FlyWire 跨性别比较二态回路。"
---

# Male CNS Connectome

**Male CNS Connectome** 是 HHMI Janelia **FlyEM Project Team** 发布的 **雄性果蝇中枢神经系统** 全连接组（**v1.0**，2026-06-08），覆盖 **中枢脑、视叶与腹神经索（VNC）** 并保留完整 **颈部连接**。这是首个 **雄性** 全 CNS 突触分辨率布线资源，与 [FlyWire](./flywire.md) 等雌性连接组共同支持 **跨性别回路比较**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CNS | Central Nervous System | 中枢神经系统 |
| VNC | Ventral Nerve Cord | 腹神经索 |
| EM | Electron Microscopy | 电子显微镜成像 |
| ROI | Region of Interest | 神经纤维网室等解剖分区 |
| CC-BY | Creative Commons Attribution | 署名即可商用的开放许可 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 规模 | v1.0 覆盖中枢脑 + 视叶 + VNC 全 CNS；**262** 种 sex-specific 与 **114** 种 sexually dimorphic 细胞类型（占中枢脑 **4.8%**）；离线连接权重表约 **1.1 GB**。 |
| 模态 | EM 体数据 + 神经元 skeleton **形态**数据 + 突触点表（`syn-points-*.feather`）+ 连接权重表 + 细胞类型注释；无 RGB / 深度等机器人传感模态。 |
| 许可证 | **CC-BY**（署名即可商用）。 |
| 适配形态 | 雄性果蝇全 CNS 回路分析与跨性别比较；非机器人本体轨迹数据。 |
| 重定向就绪度 | **不适用于运动重定向**（非运动数据）；形态学分析用 `navis` + `navis-flybrains` 在雄性/雌性模板空间变换，连接表可直接用 `pyarrow` / `pandas` 读入建图。 |

## 为什么重要

- **性别二态的首个全 CNS 参照：** 鉴定 **262** 种 sex-specific 与 **114** 种 sexually dimorphic 细胞类型（占中枢脑 **4.8%**），揭示二态性如何经连接向全脑传播。
- **脑- cord 一体：** 相较仅中枢脑的数据集，覆盖 **VNC** 使感觉-运动、行走等 **全 CNS 回路** 可端到端追踪。
- **开放与工具成熟：** CC-BY 许可；[neuPrint](./neuprint.md)、[Neuroglancer](./neuroglancer.md)、Cell Type Explorer、Clio 等 **即开即用**。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 珍妮莉亚研究园区（HHMI Janelia）FlyEM Project Team |
| 数据集 ID | `male-cns:v1.0` |
| 论文 | Berg et al., *Cell* 2026（[bioRxiv v2](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2)） |
| 许可 | CC-BY |
| 项目站 | https://male-cns.janelia.org/ |
| 下载 | https://male-cns.janelia.org/download/ |

## 流程总览

```mermaid
flowchart LR
  EM[EM 成像 + 对齐] --> PR[Proofreading v1.0]
  PR --> ANN[细胞类型 + 突触注释]
  ANN --> PUB[发布：GCS + neuPrint + Feather 表]
  PUB --> USE[探索 / 下载 / 跨性别比较]
```

## 数据访问速查

| 方式 | 入口 |
|------|------|
| Web 连接查询 | [neuPrint `male-cns:v1.0`](https://neuprint.janelia.org/?dataset=male-cns%3Av1.0) |
| 3D 浏览 | [Neuroglancer 场景](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json) |
| 细胞类型 | [Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns) |
| 雌雄比较 | [Dimorphism Explorer](https://male-cns.janelia.org/build/dimorphism_overview/) |
| Python | `neuprint-python`，`dataset='male-cns:v1.0'` |
| 原始下载 | Feather 连接表、`gs://flyem-male-cns/` 体数据 |

## 工程实践

- **程序化入门：** neuPrint 注册 → `pip install neuprint-python` → 按 [download 页](https://male-cns.janelia.org/download/) 示例查询 `fetch_neurons` / `fetch_adjacencies`。
- **离线大图：** 下载 `connectome-weights-*.feather`（~1.1 GB）与 `syn-points-*.feather`；用 `pyarrow` / `pandas` 做图分析。
- **形态学：** `navis` + `navis-flybrains` 读 skeleton 并在雄性/雌性模板空间变换。
- **与 FlyWire 对照：** Dimorphism Explorer 与 Neuroglancer 内 **共注册雌性 mesh**。

## 局限与风险

- **雄性单标本：** 个体间变异需结合 FlyWire 等多连接组统计（见 Schlegel et al. 多连接组分型工作）。
- **置信度阈值：** 下载表默认 `minconf-0.5`；高置信子集需按论文方法过滤。
- **VNC 注释成熟度：** 脑区注释较成熟；VNC 部分类型仍在社区迭代（查 release notes）。

## 关联页面

- [果蝇连接组工具栈](../concepts/fly-connectomics-stack.md)
- [FlyWire](./flywire.md) — 雌性全脑对照
- [neuPrint](./neuprint.md) — 查询服务

## 参考来源

- [Male CNS Connectome 项目站与下载页](../../sources/sites/male-cns-connectome.md)

## 推荐继续阅读

- [Janelia FlyEM Male CNS 项目页](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- [Cell 2026 论文](https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6)
