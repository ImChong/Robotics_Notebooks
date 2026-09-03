# HINT: Human-Intent Inception for Long-Horizon Robot Manipulation（arXiv:2609.02653）

> 来源归档（ingest）

- **标题：** HINT: Human-Intent Inception for Long-Horizon Robot Manipulation
- **简称：** HINT
- **类型：** paper / long-horizon-manipulation / vla / agentic-framework
- **arXiv：** <https://arxiv.org/abs/2609.02653>
- **PDF：** <https://arxiv.org/pdf/2609.02653>
- **项目页：** <https://robot-hint.github.io/> — 归档见 [`sources/sites/robot-hint.md`](../sites/robot-hint.md)
- **代码：** 截至 2026-09-03 **无** GitHub 链接
- **机构：** 浙江大学（ZJU）、上海交通大学（SJTU）、Noematrix、EndlessAI 等
- **入库日期：** 2026-09-03
- **索引来源：** [具身智能小站 8 篇盘点](../blogs/wechat_embodied_station_8_papers_open_source_2026-09-03.md)
- **一句话说明：** 在 manipulation-pattern transitions 处稀疏调用语义推理确定子任务与目标，经多视角 grounding 与视觉跟踪保持承诺；两种视觉接口（highlighting / attention-prior）不改 foundation action model 可训练参数。

## 开源状态（步骤 2.5，2026-09-03）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（方法、实验、视频） |
| GitHub / 权重 | **未见** 公开链接 |

**结论：待发布** — 可引用项目页与 arXiv；复现入口待跟进。

## 核心摘录

### 摘录 1：稀疏意图推理

- VLA 在密集动态视觉与稀疏语言下易被视觉相关性带偏。
- HINT 仅在模式切换处做语义推理，确定当前子任务与目标。
- 多视角 grounding + 视觉跟踪维持该承诺。

**对 wiki 的映射：** [paper-hint-robot-manipulation](../../wiki/entities/paper-hint-robot-manipulation.md)

### 摘录 2：双视觉接口

- image-space semantic highlighting
- attention-prior injection
- 均不向 foundation action model 引入额外可训练参数。

**对 wiki 的映射：** [paper-hint-robot-manipulation](../../wiki/entities/paper-hint-robot-manipulation.md)

### 摘录 3：实验（项目页）

- 双臂 PiPER；果蔬分拣、拼词、插孔三类长视野任务 + OOD。
- Wall-OSS-0.5 与 π₀.₅ 插件式提升意图分、子任务 SR 与全流程 SR。

**对 wiki 的映射：** [paper-hint-robot-manipulation](../../wiki/entities/paper-hint-robot-manipulation.md)

## 当前提炼状态

- [x] 项目页核查（2026-09-03）
- [x] wiki 映射：`wiki/entities/paper-hint-robot-manipulation.md`
