# Macrodata Labs（macrodata.co）

> 来源归档（site / company）

- **标题：** Macrodata Labs
- **类型：** site / company / embodied data product
- **官方入口：** <https://macrodata.co/>
- **博客：** <https://macrodata.co/blog>
- **联系 / 样例标注：** <https://macrodata.co/contact>
- **Discord：** <https://discord.gg/S8kZtmBR2x>
- **入库日期：** 2026-08-07
- **一句话说明：** Macrodata Labs 面向机器人学习提供 **具身数据智能**：把机器人与 egocentric 录像转为 **自动 QA、时间戳子任务标注** 与 **度量 3D 手部轨迹**；公开叙事强调「无需新硬件」的 RGB 管线，并对客户样例提供免费试标。

## 开源状态（步骤 2.5，截至 2026-08-07）

| 项 | 结论 |
|----|------|
| **公司专有手部跟踪 / 标注产品** | **确认未开源** — 首页与手部博客均引导 **Contact / Talk to us**；明确写明在开源组件评测之外另有 **proprietary hand-tracking** |
| **工程博客复现基线** | **部分可复现（开源组件组合）** — [Turning Egocentric Video into 3D Hand Actions](https://macrodata.co/blog/turning-egocentric-video-into-3d-hand-actions) 公开评测配方：WiLoR + HaWoR + windowed VGGT-Omega 等；**未**提供 Macrodata 自家完整仓库/权重 |
| **代码链接** | 项目页 **无** 官方 GitHub / Hugging Face 产品仓；组件需各自上游（见下方） |
| **数据** | 评测用 [HOT3D](https://huggingface.co/datasets/projectaria/hot3d)（Project Aria / Meta）；非 Macrodata 发布数据集 |

## 产品叙事（官网摘要）

1. **Timestamped subtask annotations** — 长程任务切分子任务起止；相对人工约 **20×** 成本叙事（官网口径）。
2. **3D hand-action extraction** — 估计并移除相机运动后，在稳定世界系输出每手 **21** 个度量 3D 关节。
3. **Free sample** — 发送短视频样例可获免费标注回传以评估质量。

## 关联上游开源组件（博客评测配方）

| 组件 | 角色 | 入口 |
|------|------|------|
| [WiLoR](https://github.com/rolpotamias/WiLoR) | 手部检测 | 本库 [sources/repos/wilor.md](../repos/wilor.md) |
| [HaWoR](https://github.com/ThunderVVV/HaWoR) | 相机系时序 MANO / 手重建 | 项目页 <https://hawor-project.github.io/>；本库 [sources/repos/hawor.md](../repos/hawor.md) |
| VGGT-Omega | 窗口化前馈相机轨迹 + 度量对齐 | 博客引用 Wang et al., 2026 |
| HOT3D | 光学标记真值评测集 | <https://huggingface.co/datasets/projectaria/hot3d> |

## 对 wiki 的映射

- [macrodata-egocentric-hand-action](../../wiki/methods/macrodata-egocentric-hand-action.md) — RGB-only egocentric → 度量手部动作管线（方法页）
- [WiLoR](../../wiki/methods/wilor.md) — 检测前端
- [Perceptron Egocentric](../../wiki/entities/perceptron-egocentric.md) — 同属 Macrodata **WGO** 子任务标注对照生态（语义分段轴，非本手部轨迹轴）
- [Auto-labeling Pipelines](../../wiki/methods/auto-labeling-pipelines.md) — 自动标注基础设施总览

## 当前提炼状态

- [x] 官网定位与开源边界（步骤 2.5）
- [x] 与手部动作博客交叉索引
- [ ] 若 Macrodata 后续公开完整训练/推理仓，补 `sources/repos/` 与 wiki「源码运行时序图」
