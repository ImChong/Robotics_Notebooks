---
type: entity
tags: [paper, vla, manipulation, 3dgs, real2sim2real, scene-personalization, stretch-3]
status: complete
updated: 2026-10-13
arxiv: "2610.13945"
project_id: splat2act
project: https://splat2act.github.io/
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ./paper-embodied-multimodal-grounding-3dgs-mobile-manipulation.md
  - ./paper-simfoundry-real2sim-scene-generation.md
  - ./paper-prism-real2sim2real.md
sources:
  - ../../sources/papers/splat2act_arxiv_2610_13945.md
  - ../../sources/sites/splat2act-github-io.md
summary: "Splat2Act（arXiv:2610.13945，Georgia Tech、Waymo、TRI）：以手机扫描建立楼宇级 3DGS 数字孪生和碰撞网格，由仿真 oracle 生成拾取示范，两阶段微调 π₀.₅；Stretch 3 真机 12 项任务成功率 63.9%。截至 2026-10-13，官方代码与数据仍标为即将发布。"
---

# Splat2Act：从场景扫描到个性化 VLA 操作

**Splat2Act**（*Splat2Act: From Captured Scenes to Personalized VLA Manipulation*，[arXiv:2610.13945](https://arxiv.org/abs/2610.13945)，[项目页](https://splat2act.github.io/)）把手机拍摄的室内 RGB-D 视频和用户物品照片转为仿真场景，再用仿真生成的拾取示范适配通用策略。目标是让 VLA 在特定房间里操作指定物品，而不需要在真实环境逐一遥操作采集演示。

## 一句话定义

**先把真实楼宇重建成包含外观和碰撞几何的数字孪生，再自动生成拾取任务与 oracle 示范，分两阶段将预训练 π₀.₅ 适配到目标房间和物品。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 将视觉、语言指令映射到机器人动作的策略 |
| 3DGS | 3D Gaussian Splatting | 由三维高斯表示并渲染场景外观 |
| R2S2R | Real-to-Sim-to-Real | 从真实采集重建仿真，再将训练策略迁回真机 |
| RGB-D | Red-Green-Blue plus Depth | 彩色图像与深度数据 |
| π₀.₅ | pi-zero point five | 本文适配的预训练通用操作策略 |

## 为什么重要

- **个性化的主要成本转移到环境采集。** 人只需扫描房间并拍摄物品；仿真 oracle 自动生成大批训练演示。
- **尺度从桌面扩到楼宇。** 系统对完整室内空间建立数字孪生，发现桌面以外的承载面并生成拾取任务。
- **同时报告仿真与真机结果。** Office 目标场景聚合成功率为 62.5%；同一策略在 Stretch 3 上的 12 项任务子集取得 63.9%，对应仿真子集为 66.7%。
- **展示场景适配边界。** 未适配的 π₀.₅ 在论文采集的目标场景拾取成功率为 0；在未见承载面上的成功率也明显下降，说明泛化仍依赖场景覆盖。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | Georgia Institute of Technology（佐治亚理工学院）；Waymo；Toyota Research Institute（丰田研究院） |
| **机器人平台** | Hello Robot Stretch 3 移动操作机器人 |
| **真实输入** | 手持设备采集的 RGB-D 场景视频、用户物品照片和尺寸 |
| **场景规模** | 101 个校园室内场景；中位可导航面积 104.87 m²；发现 852 个承载面 |
| **演示数据** | 68,021 条成功拾取演示，来自仿真 oracle |
| **开源状态** | **尚未公开**：截至 2026-10-13，项目页将 GitHub 代码和 Hugging Face 数据集标为 Coming soon |

## 流程总览

```mermaid
flowchart LR
  capture["手持 RGB-D 扫描 + 物品照片"]
  reconstruct["楼宇级重建：3DGS 外观 + 碰撞网格"]
  tasks["承载面发现与可行任务构造"]
  oracle["仿真几何 oracle：规划、执行、重试"]
  demos["成功拾取演示"]
  adapt["两阶段微调 π₀.₅：广泛适配 → 场景个性化"]
  real["Stretch 3 真机拾取"]
  capture --> reconstruct --> tasks --> oracle --> demos --> adapt --> real
```

## 方法细节

### 1. 将扫描结果变成可执行场景

项目页描述的流程从约 20–30 分钟手持 RGB-D 扫描开始。3D Gaussian Splatting（3DGS）保存真实空间的视觉外观，场景网格提供碰撞和任务规划所需几何。论文使用 DN-Splatter 做深度与法线正则化，并用 SuGaR 从高斯场提取表面网格。物品侧可从公开三维资产选择，也可根据物品照片和实测尺寸创建可移动资产。

### 2. 自动发现承载面并生成任务

系统围绕机器人几何建立可导航空间，从朝上的几何表面中发现稳定、可触及的承载面（receptacles）。任务生成器采样物品摆放、机器人初始位姿和干扰物，并过滤不可达配置。使用特权几何状态的 oracle 负责规划、执行与失败重试，产出成功拾取轨迹；训练示范不来自用户遥操作。

### 3. 两阶段适配 π₀.₅

- **Stage 1：广泛适配。** 用数百种公开物品的演示让策略先适应 Stretch 机器人与拾取任务。数据可来自目标场景，也可合并 101 个场景。
- **Stage 2：目标个性化。** 再用用户自己的物品和目标场景承载面演示做场景专属微调。
- 部署策略接收头部与腕部 RGB、机器人本体状态和语言提示。几何 oracle 用于生成训练示范，不是部署策略的输入。

## 源码运行时序图

**不适用（截至 2026-10-13）**：项目页将 GitHub 代码和 Hugging Face 数据集标为 Coming soon，目前没有公开可核对的训练、仿真或部署入口。论文给出了系统阶段，但无法据此画出与实际仓库文件对齐的源码时序图。

## 实验与评测

| 评测 | 结果 | 读法 |
|------|------:|------|
| 未适配 π₀.₅ 的目标场景拾取 | 0% | 该场景与 Stretch 动作空间未见于其原始策略经验 |
| Office 目标场景，广泛适配后再个性化 | 62.5% | 项目页报告比从预训练模型直接个性化高 17.5 个百分点 |
| 同一 12 项任务子集，仿真 | 66.7% | 12 个仿真 episodes |
| 同一 12 项任务子集，Stretch 3 真机 | 63.9% | 36 次真实尝试；与仿真结果接近 |
| 场景集合 | 101 个扫描环境 | 中位可导航面积 104.87 m²，平均约 7 个承载面 |
| 自动生成数据 | 852 个承载面；68,021 条成功演示 | 演示由仿真 oracle 生成 |

论文的 62.5% 是目标场景聚合评测，63.9% 则来自固定的 12 项真机任务子集；两者分母与协议不同。项目页还指出，在未见实验室承载面上的成功率明显下降。因此结果支持的是**已扫描场景内的个性化迁移**，不能解读为任意未见家具都能直接泛化。长时程移动操作仍是后续方向。

## 结论

**Splat2Act 将场景数字化、自动生成仿真拾取示范和 VLA 个性化串成闭环；它减少真实演示采集，但当前验证主要支持已扫描场景内的拾取。**

1. **可复用之处是数据生成闭环。** 扫描场景和物品资产被转成大量仿真任务，减少逐环境遥操作。
2. **个性化需要两层数据。** 先适应机器人与任务，再适应目标场景和物品；Office 中完整两阶段方法比直接个性化高 17.5 个百分点。
3. **真机迁移值得关注。** 固定 12 项任务子集中，真机成功率 63.9%，仿真为 66.7%。
4. **读数要区分覆盖范围。** 未见承载面表现下降，结果不等于开放世界泛化。
5. **复现暂受开源状态限制。** 项目页仍标注代码和数据即将发布，当前不能独立复跑完整管线。

## 局限与风险

- **数据分布受场景采集约束：** 个性化策略主要围绕已扫描场景及其承载面；离开这些场景后表现会下降。
- **任务范围较窄：** 论文聚焦 receptacle-local pickup；较长程移动与连续操作仍待验证。
- **依赖特权几何 oracle：** 训练演示由仿真几何状态生成，完整复现需实现重建、碰撞几何、可行性筛选和 oracle 规划。
- **开源未落地：** 截至入库日项目页未提供可访问代码仓库或数据集链接；公开后应补充源码归档与运行时序图。

## 关联页面

- [VLA 方法页](../methods/vla.md) — 预训练策略与任务个性化背景
- [Manipulation 操作任务](../tasks/manipulation.md) — 本文属于场景化物体拾取
- [Embodied MM Grounding（3DGS）](./paper-embodied-multimodal-grounding-3dgs-mobile-manipulation.md) — 以 3DGS 场景表示支持移动操作
- [SimFoundry](./paper-simfoundry-real2sim-scene-generation.md) — 对照真实场景转仿真资产与任务生成路线
- [PRISM Real2Sim2Real](./paper-prism-real2sim2real.md) — 视频驱动仿真示范生成与真机迁移

## 参考来源

- [Splat2Act 论文摘录](../../sources/papers/splat2act_arxiv_2610_13945.md)
- [Splat2Act 项目页核查](../../sources/sites/splat2act-github-io.md)
- [arXiv:2610.13945](https://arxiv.org/abs/2610.13945)

## 推荐继续阅读

- [Splat2Act 项目页](https://splat2act.github.io/)
- [Splat2Act 论文](https://arxiv.org/abs/2610.13945)
- [VLA 方法页](../methods/vla.md)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
