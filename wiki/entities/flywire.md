---
type: entity
tags: [connectomics, dataset, drosophila, flywire, princeton, neuroscience, community]
status: complete
updated: 2026-09-10
related:
  - ../concepts/fly-connectomics-stack.md
  - ./male-cns-connectome.md
  - ./neuroglancer.md
  - ./neuprint.md
  - ./flybrainlab.md
sources:
  - ../../sources/sites/flywire.md
summary: "FlyWire Consortium 构建的雌性果蝇全脑连接组平台：~140K proofread 神经元、50M+ 突触，Nature 2024 旗舰论文，Codex 探索入口。"
---

# FlyWire

**FlyWire**（https://flywire.ai/）是 **FlyWire Consortium** 经大规模专家 proofreading 完成的 **雌性成年果蝇全脑** 连接组平台。截至 Nature 2024 旗舰论文，含 **139,255** 个 proofread 神经元、**50M+** 突触（含神经递质）与 **100K+** 社区细胞注释，是 [Male CNS Connectome](./male-cns-connectome.md) 的 **雌性对照** 与跨性别比较基准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EM | Electron Microscopy | 电子显微镜成像 |
| CNS | Central Nervous System | 中枢神经系统 |
| NIH | National Institutes of Health | 美国国立卫生研究院资助方 |
| MRC | Medical Research Council | 英国医学研究理事会 |
| API | Application Programming Interface | Codex 等程序化接口 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 规模 | **139,255** proofread 神经元、**50M+** 突触、**100K+** 社区 cell labels（Nature 2024 快照）。 |
| 模态 | EM 体数据 + 神经元分割 mesh / skeleton **形态**数据 + 突触点表（含神经递质预测）+ 文本型细胞类型注释；无 RGB / 深度等机器人传感模态。 |
| 许可证 | 连接组数据经 Codex 开放探索与程序化获取；注释仓库 `flywire_annotations` 在 GitHub 开源；平台源码非单一公开 monorepo，引用口径以官网 citation 指南为准。 |
| 适配形态 | 雌性果蝇全脑回路分析与类脑建模输入；非机器人本体轨迹数据。 |
| 重定向就绪度 | **不适用于运动重定向**（非运动数据）；跨性别复用须走 Male CNS 的 Dimorphism Explorer / Neuroglancer **共注册**场景，不可直接横比统计。 |

## 为什么重要

- **首个全脑成虫连接组：** 把果蝇脑从「区域统计」推进到 **逐神经元、逐突触** 可查。
- **社区驱动 curation：** 数百科学家参与 proofreading 与注释，形成持续更新的 **细胞类型目录**。
- **工具链外溢：** Google Research 3D 查看器、突触预测与 [Neuroglancer](./neuroglancer.md) 生态直接惠及后续 Male CNS 发布。

## 核心信息

| 字段 | 内容 |
|------|------|
| 官网 | https://flywire.ai/ |
| 探索 | [Codex](https://codex.flywire.ai/) |
| 旗舰论文 | Dorkenwald et al., *Nature* 2024 |
| 注释仓库 | [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) |
| 性别 | **雌性** 全脑（不含完整 VNC 一体发布） |

## 流程总览

```mermaid
flowchart LR
  IMG[Janelia EM 成像] --> AUTO[Princeton 自动重建]
  AUTO --> FW[FlyWire 平台 proofreading]
  FW --> SYN[突触 + 神经递质]
  SYN --> CODEX[Codex 发布与社区注释]
```

## 工程实践

- **浏览：** 打开 Codex 按细胞类型或连接搜索；3D 视图基于 Neuroglancer 技术栈。
- **与 Male CNS 比较：** 使用 Male CNS **Dimorphism Explorer** 或共注册 Neuroglancer 场景。
- **程序化：** Consortium 提供的 Python/R 工具与 [neuPrint](./neuprint.md) 部分数据集互通；注释更新跟踪 `flywire_annotations` 仓库。

## 局限与风险

- **雌性 vs 雄性：** 行为与回路二态研究须明确对比 [Male CNS](./male-cns-connectome.md)，不可混用统计。
- **平台源码：** 非单一开源 monorepo；复现 UI 需组合 Neuroglancer + 自建数据服务。
- **注释版本：** 社区注释持续更新；论文复现应锁定发布快照与 citation 指南。

## 关联页面

- [Male CNS Connectome](./male-cns-connectome.md)
- [FlyBrainLab](./flybrainlab.md)
- [果蝇连接组工具栈](../concepts/fly-connectomics-stack.md)

## 参考来源

- [FlyWire 官网归档](../../sources/sites/flywire.md)

## 推荐继续阅读

- [FlyWire Nature 2024 论文](https://flywire.ai/)
- [Schlegel et al. 多连接组细胞分型](https://www.nature.com/articles/s41586-024-07686-4)
