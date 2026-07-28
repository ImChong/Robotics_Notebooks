---
type: entity
tags: [site, project, semantic-mapping, open-vocabulary, sam, tum, eth]
status: complete
updated: 2026-07-26
related:
  - ../queries/robot-perception-stack-selection-loop.md
  - ../queries/go2-3d-semantic-mapping-sam-pipeline.md
  - ./dualmap.md
  - ./ovo-semantic-mapping.md
  - ./ov-sam3d.md
  - ./autonomy-stack-go2.md
sources:
  - ../../sources/sites/findanything.md
summary: "FindAnything 项目页：对象级开放词汇体素子地图，强调机载实时（Jetson Orin NX 演示）；代码宣称将并入 OKVIS2-X，截至入库日尚无独立开源仓实体。"
---

# FindAnything

**FindAnything**（[项目页](https://ethz-mrl.github.io/findanything/)，arXiv:2504.08603）是面向机器人探索的 **开放词汇、对象中心** 三维建图项目（TUM × ETH）。

## 一句话定义

把 SAM/eSAM 二维区域与视觉语言特征聚合进 **对象级体素子地图**，在几何重建之外支持自然语言查询，并强调大场景内存可扩展与资源受限平台机载运行。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SAM | Segment Anything Model | 二维分割；项目叙述含 eSAM 变体 |
| CLIP | Contrastive Language–Image Pretraining | 开放词汇查询常用视觉语言特征族 |
| TSDF | Truncated Signed Distance Function | 稠密几何/体素表示相关 |
| MAV | Micro Aerial Vehicle | 项目页演示载体之一 |
| OKVIS | Open Keyframe-based Visual-Inertial SLAM | 代码宣称将并入的 OKVIS2-X 宿主系 |
| OV | Open-Vocabulary | 不限定闭集类别的语义查询 |

## 为什么重要

- 与 [DualMap](./dualmap.md) / [OVO](./ovo-semantic-mapping.md) 同属「在线开放词汇语义地图」选型池，但强调 **对象级体素子地图 + 机载算力**。
- 项目页已展示 **Jetson Orin NX** 级部署与语言引导探索，对 GO2 机载语义有参考价值。
- 当前以 **项目页实体** 占位；开源仓落地后再补仓库地址与运行时分析。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 慕尼黑工业大学（TU Munich）；苏黎世联邦理工（ETH Zürich） |
| 项目页 | <https://ethz-mrl.github.io/findanything/> |
| 论文 | <https://arxiv.org/abs/2504.08603> |
| 开源（截至 2026-07-26） | **宣称将开源 / 待并入宿主仓**：代码将作为 **OKVIS2-X** 附加功能发布；**尚无**独立可 pip 安装的 FindAnything 单仓 |
| 代码仓（待补） | *待官方发布后填写 URL，并新建 `sources/repos/` + 本页「源码分析」节* |

## 核心原理（项目页级）

| 要点 | 说明 |
|------|------|
| 几何 | 稠密体素子地图，可支持回环等漂移修正叙事 |
| 语义 | 像素级视觉语言特征按 eSAM 段聚合到 **对象** |
| 查询 | 自然语言 ↔ 三维几何映射，偏探索/搜救类下游 |

## 工程实践

1. **现阶段：** 读项目页与 arXiv，对照本库 [GO2 语义 Query](../queries/go2-3d-semantic-mapping-sam-pipeline.md) 选型；**不要**假设已有可 clone 训练/部署仓。
2. **跟进开源：** 定期打开 OKVIS2-X / 项目页 Resources；一旦有稳定 GitHub：
   - 新建 `sources/repos/findanything.md`（或宿主仓条目）
   - 更新本页「代码仓」行与开源状态
   - 补「源码运行时序图 / README 入口分析」
3. **接到 GO2：** 仍建议先 Point-LIO 几何锐利，再评估机载语义帧率。

## 局限与风险

- **代码未独立公开**：复现与选型排期须按「待发布」管理。
- 宿主仓合入节奏未知；功能边界以最终 README 为准。
- 与 CMU 几何栈 [autonomy_stack_go2](./autonomy-stack-go2.md) 无关，勿混为同一「CMU 工作」。

## 关联页面

- [Query：机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 本页属**第③层 2D→3D 提升与语义建图**（对象级开放词汇 3D 语义建图）
- [GO2 三维语义建图与 SAM 流水线](../queries/go2-3d-semantic-mapping-sam-pipeline.md)
- [DualMap](./dualmap.md)
- [OVO](./ovo-semantic-mapping.md)
- [OV-SAM3D](./ov-sam3d.md)
- [autonomy_stack_go2](./autonomy-stack-go2.md)
- [CMU MSCV Semantic 3D Mapping](./cmu-mscv-semantic-3d-mapping.md)

## 参考来源

- [sources/sites/findanything.md](../../sources/sites/findanything.md)
- 项目页：<https://ethz-mrl.github.io/findanything/>
- arXiv：<https://arxiv.org/abs/2504.08603>

## 推荐继续阅读

- DualMap 项目页（已开源对照）：<https://eku127.github.io/DualMap/>
