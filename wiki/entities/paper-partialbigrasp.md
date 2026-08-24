---
type: entity
tags: [paper, bimanual-grasping, 3d-perception, partial-point-cloud, force-closure]
status: complete
updated: 2026-08-24
arxiv: "2608.19188"
related:
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
  - ./paper-real-bi-dex-grasp.md
  - ./paper-mango-grasp.md
  - ../queries/robot-perception-stack-selection-loop.md
  - ../overview/vla-predict-grasp-9-papers-technology-map.md
sources:
  - ../../sources/papers/partialbigrasp_arxiv_2608_19188.md
  - ../../sources/sites/partialbigrasp-github-io.md
  - ../../sources/repos/partialbigrasp-codebase.md
  - ../../sources/blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md
  - ../../sources/blogs/wechat_embodied_station_9_papers_vla_predict_grasp_2026-08-24.md
summary: "PartialBiGrasp（arXiv:2608.19188，IIIT Hyderabad）：局部点云→占据网络补隐藏几何→力闭合双臂抓取对；DG16M ~55% FC。架构仓部分开源，权重/训练 TODO。"
---

# PartialBiGrasp：残缺观测下的双臂抓取局部几何补全

**PartialBiGrasp**（*Inferring Hidden Local Geometry for Bimanual Grasping from Partial Views*；[arXiv:2608.19188](https://arxiv.org/abs/2608.19188)，[项目页](https://partialbigrasp.github.io/)）由 **IIIT Hyderabad** 提出：大型/几何复杂物体在真实 RGB-D 下往往只有 **局部点云**，完整重建既慢又常与接触决策无关。

## 一句话定义

**不重建完整物体，只用卷积占据网络从局部观测推断厚度、边缘与夹爪间隙，再采样优化出力闭合双臂抓取对。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FC | Force Closure | 力闭合抓取约束 |
| RGB-D | RGB + Depth | 彩色深度传感 |
| DG16M | DexGraspNet 16M 子集 | 大规模仿真评测集 |
| CON | Convolutional Occupancy Network | 隐式局部几何网络 |
| BiGrasp | Bimanual Grasping | 双臂协同抓取 |

## 为什么重要

- 双臂协同常面对 **单视角遮挡**：背面厚度与第二夹爪间隙不可见。
- 完整 mesh 重建误差会传播到 grasp planner；**局部几何** 才是力闭合判据所需。
- 文内综述将其归入「补全隐藏状态」闭环：先修复感知再决策抓取。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | IIIT Hyderabad（Robotics Research Center） |
| **传感** | 局部点云（RealSense D455 等 RGB-D） |
| **输出** | 满足 FC 的双臂 grasp pair |
| **开源** | **部分开源**（[codebase](https://github.com/partialbigrasp/codebase) 架构已发；权重/训练/数据 **TODO**） |

## 核心原理

### 管线

```mermaid
flowchart LR
  pc["局部点云"]
  occ["卷积占据网络\n厚度/边缘/间隙"]
  single["单臂 grasp 候选"]
  pair["FC pairing critic"]
  refine["占据引导采样 refinement"]
  out["双臂 FC 抓取对"]
  pc --> occ --> single --> pair --> refine --> out
```

1. **占据网络** — 全局+局部编码器隐式预测可抓性、无碰撞接触区、物体厚度。
2. **单臂生成** — 各臂独立候选 grasp。
3. **FC pairing** — critic 筛选满足力闭合的双臂组合。
4. **采样 refinement** — 用局部占据修正不完整几何带来的歧义。

## 源码运行时序图

**不适用** — 截至 **2026-08-21** [codebase](https://github.com/partialbigrasp/codebase) 标注 in progress，无权重/推理 notebook/环境脚本。发布后预期：点云预处理 → 占据推理 → pairing + refinement → 仿真/实机执行。

## 工程实践

| 项 | 建议 |
|----|------|
| 何时引用 | 双臂 + **partial view** + 需 FC 保证，而非单臂 top-down grasp |
| 与完整重建对比 | 若只需接触区几何，优先局部占据而非全局 mesh |
| 实机 | 11 物体 RealSense 集验证；迁移到新物体需查 DG16M 覆盖域 |
| 复现 | 跟踪 codebase README TODO 清单 |

## 实验与评测

- **仿真：** DG16M 上 ~**55%** FC vs baseline ~**22%**。
- **实机：** 11 物体 RealSense D455 新物体实验（项目页/PDF 细节）。
- **指标：** 解析 grasp 指标 + FC 成功率。

## 结论

**双臂抓取正从「先重建完整物体」转向「只补全与接触有关的局部几何」。**

1. **局部占据** — 隐藏厚度/边缘是 FC 判据的关键，不必 full mesh。
2. **FC + refinement** — pairing critic 与采样优化共同消化 partial view 歧义。
3. **仿真增益大** — DG16M FC 率约为 baseline 2.5×。
4. **开源边界** — 架构可读，完整复现等权重与训练栈。
5. **系统读法** — 与 LT-Mem / Hydra-0 同属「补全隐藏状态」能力线。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 先做完整 mesh 重建再规划 | 重建慢，且误差会整体传播到 grasp planner；PartialBiGrasp 只补 **与接触有关的局部几何**（厚度/边缘/夹爪间隙） |
| DG16M 上的 baseline | FC 成功率 ~**55% vs ~22%**，约 2.5× |
| [Real Bi-Dex Grasp](./paper-real-bi-dex-grasp.md) | 另一条双臂抓取路线；本页的分工点是「**partial view** 下先补隐藏几何再配对」 |
| [Mango Grasp](./paper-mango-grasp.md) | 抓取规划侧对照；本页把力闭合判据前移到占据网络推断出的局部几何上 |
| 单臂 top-down grasp pipeline | 计算与规划复杂度都更低；只有「双臂 + partial view + 需 FC 保证」三者同时成立时才值得上本方法 |
| [Hydra-0](./paper-hydra-0.md) / [LT-Mem](./paper-lt-mem.md) | 综述同批「补全隐藏状态」能力线：Hydra 补的是未来帧、LT-Mem 补的是历史记忆，本页补的是**当前观测里看不见的几何** |

## 局限与风险

- **部分开源** — 权重、数据集、推理路径未发布。
- **对象域** — 大/重/几何复杂物体为主；小物体或极端遮挡需另验证。
- **双臂标定** — 双视角/双臂外参误差会放大局部几何误差。
- **与单臂 pipeline 对比** — 计算与规划复杂度高于单 gripper top-down。

## 关联页面

- [Manipulation 任务](../tasks/manipulation.md)
- [Real Bi-Dex Grasp](./paper-real-bi-dex-grasp.md) — 另一双臂抓取路线
- [Mango Grasp](./paper-mango-grasp.md) — 抓取规划对照
- [Hydra-0](./paper-hydra-0.md) — 综述同批「补全隐藏状态」
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 本页落其 ③ 层 2D→3D 提升：不建完整语义地图，只从局部点云补出力闭合判据需要的接触区几何，再交 ④ 层抓取规划消费
- [VLA·预测·抓取 9 篇技术地图](../overview/vla-predict-grasp-9-papers-technology-map.md)

## 参考来源

- [PartialBiGrasp 论文归档](../../sources/papers/partialbigrasp_arxiv_2608_19188.md)
- [partialbigrasp 项目页](../../sources/sites/partialbigrasp-github-io.md)
- [partialbigrasp/codebase 归档](../../sources/repos/partialbigrasp-codebase.md)
- [具身智能小站 8 篇综述](../../sources/blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md)
- [具身智能小站 9 篇盘点（2026-08-24）](../../sources/blogs/wechat_embodied_station_9_papers_vla_predict_grasp_2026-08-24.md)

## 推荐继续阅读

- [arXiv:2608.19188 PDF](https://arxiv.org/pdf/2608.19188)
- [PartialBiGrasp 项目页](https://partialbigrasp.github.io/)
