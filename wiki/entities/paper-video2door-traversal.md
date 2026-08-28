---
type: entity
tags:
  - paper
  - loco-manipulation
  - mobile-manipulation
  - real-to-sim-to-real
  - door-traversal
  - imitation-learning
  - action-chunking
  - articulated-objects
  - isaac-gym
  - unitree
  - sjtu
  - sdu
  - neowa
status: complete
updated: 2026-08-28
arxiv: "2608.20251"
related:
  - ../tasks/loco-manipulation.md
  - ../concepts/sim2real.md
  - ../methods/action-chunking.md
  - ./articraft.md
  - ./physx-omni.md
  - ./paper-doorman-opening-sim2real-door.md
  - ./paper-agentic-real2sim.md
  - ./isaac-gym-isaac-lab.md
  - ./paper-smpc2rl-loco-manipulation.md
  - ../tasks/manipulation.md
  - ../overview/video-contact-control-10-papers-technology-map.md
sources:
  - ../../sources/papers/video2door_traversal_arxiv_2608_20251.md
  - ../../sources/sites/video2door-traversal.md
  - ../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md
summary: "Video2DoorTraversal（arXiv:2608.20251，SJTU×SDU×NeoWa）：单 RGB 视频经 DoorTwin 建成实例对齐可仿真门孪生，仿真闭环 agent 生成穿门演示，ArticuACT 以双深度 + 机器人系 Plücker 学底盘–臂协同；A2-W 五扇真门 96.57%、相近未见门 80.95%；代码待发布。"
---

# Video2DoorTraversal

**Video2DoorTraversal**（*Push Door Traversal via Simulated Door Twins*，[arXiv:2608.20251](https://arxiv.org/abs/2608.20251)，[项目页](https://video2doortraversal.github.io/)）由 **上海交通大学 × 山东大学 × 纽娲机器人（NeoWa Robotics）** 提出：把「看一次真实门」写成可反复试错的仿真资产，再在轮足移动操作平台上学习闭环推门穿越。对应作者为 Tang、Chen、Xie、Li、Shu、Jiang、Hu、Li、Zhang、Song、Yang。

## 一句话定义

**一段手持 RGB 视频重建实例对齐的可仿真门孪生，仿真里生成可执行穿门演示，再用双深度 ACT 在真机上闭环走完接近–开锁–推门–过门洞。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoorTwin | Door Twin | 本文从单视频重建的实例对齐、带关节、可仿真门资产 |
| ArticuACT | Articulated Action Chunking Transformer | 本文策略：双深度 ACT + 机器人系 Plücker + 交互进度辅助头 |
| ACT | Action Chunking Transformer | ArticuACT 的模仿学习骨干，一次预测动作块 |
| R2S2R | Real-to-Sim-to-Real | 真机观测 → 仿真资产/数据 → 真机策略 |
| VLM | Vision-Language Model | DoorTwin 选参考帧与 visual critic；专家修复也用 VLM |
| DR | Domain Randomization | 专家程序验证后，对位姿/摩擦/相机/深度噪声做随机化 |
| IL | Imitation Learning | 本文主策略路线，对照 DoorGym 等 RL 开门基线 |
| PCA | Principal Component Analysis | 从聚合点云估计门宽、高与法向 |
| A2-W | Unitree A2-W | 宇树轮足底座，本文搭 Z1 臂做真机 |

## 为什么重要

- **把「这扇门」变成可迭代试错的仿真对象：** 多数开门/穿门工作用程序化或预建资产（[DoorMan](./paper-doorman-opening-sim2real-door.md)、UniDoorManip、腿式 teacher–student）。这里用一次视频把度量尺度、铰链侧、把手位置和外观焊进同一孪生，后续专家生成与策略训练不再依赖人工建模。
- **穿门不是开门：** 仿真里 UniDoorManip 开门 74.22% 但穿越掉到 50.78%——开到 80° 之后还要底盘–臂协同过窄门洞。本文把完整穿越当作主指标。
- **部署契约清楚：** 训练可用特权交互标注，上机只吃头/腕深度 + 本体；感知与策略都在机载（Jetson Orin NX）。
- **阅读坐标：** 收录于 [视频–接触–控制 10 篇技术地图](../overview/video-contact-control-10-papers-technology-map.md) 的「仿真与控制上机」组。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 上海交通大学（SJTU）；山东大学（SDU）；纽娲机器人（NeoWa Robotics） |
| **出处** | arXiv:2608.20251（2026-08，v1） |
| **平台** | Unitree A2-W 轮足 + Unitree Z1 臂；头/腕 RealSense D435 |
| **仿真** | Isaac Gym，专家 50 Hz，策略数据 25 Hz |
| **策略** | ArticuACT：chunk \(H=100\)，动作 9 维（底盘 \(v_x,\omega_z\) + 六臂关节 + 夹爪） |
| **数据** | 每扇门 200 条成功仿真演示；深度裁到 \([0.2,1.5]\) m |
| **开源（截至 2026-08-28）** | **待发布**：项目页 **Code Coming soon**，无 GitHub / 权重链 |

## 核心原理

### 流程总览

```mermaid
flowchart LR
  vid[单段手持 RGB 视频] --> twin[DoorTwin 关节门孪生]
  twin --> agent[仿真闭环技能程序]
  agent --> demo[域随机化成功演示]
  demo --> pol[ArticuACT 训练]
  pol --> real[A2-W 机载双深度闭环]
```

三块共用同一扇门的关节表示 \(\Theta_D\)：重建给出几何与关节，专家在其上搜可执行程序，策略在其上做接触丰富的模仿。

### DoorTwin：单视频 → 可仿真关节资产

1. **度量几何接地：** DAGE 估度量深度与相机位姿；玻璃/反光再走 LingBot-Depth。SAM 3 掩膜后把多帧点云变到门坐标系，PCA 定宽、高、法向。
2. **板相对把手：** 参考帧上分割门板与把手，把手中心相对门板系写成 \(\Delta\mathbf{p}_h\)，与全局尺寸一起作为 Articraft 约束。
3. **粗到细程序生成：** 框固定、门板 revolute、把手贴接地位置；先大门体再修小把手，减轻尺度差。Articraft 原有校验只保证结构可加载。
4. **参考视角 critic：** 用参考帧内参把资产渲染回去，VLM 评轮廓、把手类型、铰链侧、比例、把手位置；失败则回写生成 agent，度量约束保持不变。
5. **去光贴图：** 几何锁定后 GPT 去阴影，Tripo 3D 生成纹理。程序化材质往往比真门外观简单，这一步是为 sim-to-real 视觉对齐，不是再改关节。

直觉：Articraft 能写出「一扇能仿真的门」，但不能保证「就是视频里那扇」；DoorTwin 用视频度量把绝对尺度和小零件位置钉死，再用渲染对照补实例相似。

### Agentic 专家：技能程序而非低层控制器代码

技能集 \(\mathcal{S}\)：BaseMoveTo、EE_Approach、Close_Gripper、Rotate_Handle、Push_Door、Pass、ReleaseAndRetract。每个技能只带接近距离、抓取偏置、把手转角、接触偏置、底盘速度、相位时长等紧凑参数；交互目标写在把手局部系，推门方向由铰链朝向决定。

失败诊断走 generate–execute–diagnose–refine：结构化进度（把手转角、门角、解锁、机体通过）+ 可行性信号 + 多视角关键帧 → 有界改参数 → 局部仿真搜索，只接受满足任务/碰撞/运动学约束的候选。没有仿真 rollout 时成功率从 85.63% 掉到 48.13%，说明「几何对齐或纯 VLM 改日志」不够产生物理可执行轨迹。

### ArticuACT：几何接地 + 交互进度的 ACT

相对 vanilla ACT：

- **机器人系 Plücker：** 头/腕像素射线都变到基座系，给像素与动作坐标系显式对应，减轻双相机视点歧义。
- **未来交互状态：** 每步预测接触、把手进度、开门进度；只作辅助损失，不进入动作接口——部署仍是 9 维命令。
- **命令空间：** 同模块下关节命令优于末端命令；作者归因于更贴近低层伺服、接触阶段更稳。

## 源码运行时序图

**不适用。** 截至 **2026-08-28**，[项目页](https://video2doortraversal.github.io/) 仅标注 **Code Coming soon**，未列可辨识的训练/推理仓库；无法对齐 `sources/repos/` 入口绘制可复现运行时序。开放后应补 `sources/repos/` 并补本图。论文描述的逻辑顺序是：视频 → DoorTwin（DAGE / SAM 3 / Articraft / Tripo）→ Isaac Gym 专家采集 → ArticuACT 训练 → Orin NX 机载推理。

## 工程实践

| 项 | 建议 |
|----|------|
| 选型定位 | 目标是 **某扇真实推门的完整穿越**，且能接受「先建孪生再仿真正则化」，而不是从程序化门资产直接 RL。若已是人形 RGB 开门、要零样本未见门，对照 [DoorMan](./paper-doorman-opening-sim2real-door.md)。 |
| 采集 | 手机一段能看全门的 RGB 即可；玻璃门要准备深度修补。不需要额外物体扫描。 |
| 专家环 | 先验证一条可执行程序，再 DR 重放；只留扰动下仍成功的轨迹。去掉仿真反馈会大幅掉成功率。 |
| 策略输入 | 双深度 + 9 维本体；深度裁剪与噪声增强按论文 \([0.2,1.5]\) m。chunk \(H=100\) @ 25 Hz。 |
| 上机 | 低层底盘在 A2-W 机载电脑；视觉编码与策略在 Jetson Orin NX；不要默认「整网都能在底盘 MCU 上跑」。 |
| 复现入口 | **不适用**（官方代码待发布）。上游可参考 [Articraft](./articraft.md) 与 [ACT](../methods/action-chunking.md)，但不能复现本文度量接地与 critic 回路。 |

## 实验与评测

### 资产生成（20 扇真门）

| 方法 | SS ↑ | mIoU ↑ | PSNR ↑ | VLM Score ↑ |
|------|------|--------|--------|-------------|
| PhysX-Omni | 65.75 | 0.635 | 17.51 | 11.54 |
| Articraft | 89.03 | 0.880 | 16.64 | 28.75 |
| Articulate-Anything | 86.62 | 0.831 | 16.03 | 11.59 |
| DoorTwin | **94.95** | **0.972** | **18.53** | **56.74** |

PhysX-Omni 会把门认成别的关节物；Articulate-Anything 门型对、比例和把手位置常偏；Articraft 结构合法但尺度与外观仍有可见误差。DoorTwin 的 SSIM 略低于 PhysX-Omni，作者归因于局部外观差对 SSIM 更敏感。

### 仿真穿门（20 扇门，256 trial）

| 方法 | 开门成功率 | 穿越成功率 |
|------|------------|------------|
| Replay / OA Replay | 14.06% / 53.13% | 同左（开环或按把手位姿整体 warp） |
| DoorGym PPO | 12.50% | 12.50% |
| UniDoorManip | 74.22% | 50.78% |
| Vanilla ACT / DP / DP3 | 64–68% | 64–66% |
| ArticuACT | **98.44%** | **97.27%** |

穿越判定：先解锁并门角 \(>80^\circ\)（20 s 内），再要求底盘越过门洞至少 1 m。开锁前底盘撞门或超时算失败。

### 真机

- 五扇训练门：35/35、32/35、35/35、33/35、34/35，合计 **169/175 = 96.57%**；全程约 **13 s**。
- Vanilla ACT 真机平均 **65.71%**；开环 Replay **18.29%**。
- 结构相近未见三扇、无再生成轨迹、无真机微调：**25/35、31/35、29/35**，平均 **80.95%**。
- 交互进度头在真机上与接触 / 转把手 / 开门相位对齐，底盘前向速度在穿越段平滑抬升。

## 结论

**穿门的关键不是再堆一个端到端黑盒，而是把一次观察变成度量对齐、可仿真、可诊断失败的门孪生，再用几何条件化的动作块策略吃机载深度。**

1. **实例孪生比通用门资产值钱** — DoorTwin 相对 Articraft 的增益主要在尺度、把手位置和实例外观，而不是「会不会写 revolute」。
2. **专家必须过仿真物理** — 去掉 rollout 反馈，技能程序成功率接近腰斩；VLM 改日志补不齐接触与碰撞。
3. **读真机数字时看「训练门 vs 相近未见门」** — 96.57% 是五扇已建孪生的门；80.95% 才是结构相近 zero-shot。跨推/拉、闭门器、完全不同几何未给出。
4. **关节命令 + Plücker + 交互辅助头是互补的** — 单独加模块都有收益，组合在关节空间最好；辅助头不上真机控制接口。
5. **代码未开** — 工程复现目前只能对照上游 Articraft / ACT / Isaac Gym，不能当作可跑通的官方栈。

## 与其他工作对比

| 路线 | 输入 | 本体 | 开门 vs 穿越 | 专家 | 部署观测 |
|------|------|------|--------------|------|----------|
| Human2Sim2Robot / X-SIM | 人视频 + 扫描 | 固定基座 | 无穿越 | 否 | 位姿 / RGB |
| UniDoorManip | 预建资产 | 移动 | 主攻开门 | 否 | 点云 |
| [DoorMan](./paper-doorman-opening-sim2real-door.md) | 程序化门 + 视觉 DR | 人形 G1 | 开/关/穿 | 特权 PPO 教师 | 机载 RGB |
| 本文 | **单 RGB 视频** | 轮足 A2-W + 臂 | **完整穿越** | 仿真闭环技能程序 | 机载双深度 |

与 [Agentic Real2Sim](./paper-agentic-real2sim.md) 同属「VLM + 仿真闭环」，但单位不同：那里是 DROID **episode twin** 回放；这里是 **单门资产孪生** 再学穿门策略。

## 局限与风险

- **只覆盖推门。** 结论写明未来才做拉门与更多机构；闭门器、弹簧回弹、双向人流未作为主评测。
- **zero-shot 口径是「结构相近」。** 不要读成任意未见门或跨楼宇泛化。
- **深度工作距离短。** 裁到 1.5 m，适合近门交互，不适合远距导航开门。
- **管线依赖闭源/第三方服务。** Articraft 生成、GPT 去光、Tripo 贴图、VLM critic 都不是可离线复现的单一开源仓。
- **开源待发布。** 截至 2026-08-28 项目页无代码；指标不可独立复核。
- **本体特化。** 9 维动作绑 A2-W + Z1；换人形或纯轮式需重做专家程序与命令空间。

## 关联页面

- [Loco-Manipulation](../tasks/loco-manipulation.md) — 边移动边操作总任务页；本工作是轮足穿门样本。
- [Sim2Real](../concepts/sim2real.md) — R2S2R 迁移总图。
- [Action Chunking](../methods/action-chunking.md) — ArticuACT 的 ACT 骨干与 chunk 部署语境。
- [Articraft](./articraft.md) — DoorTwin 调用的程序化关节资产生成。
- [PhysX-Omni](./physx-omni.md) — 门资产生成对照基线。
- [DoorMan](./paper-doorman-opening-sim2real-door.md) — 人形纯 RGB 开门：程序化资产 + 特权教师，而非单视频孪生。
- [Agentic Real2Sim](./paper-agentic-real2sim.md) — simulator-in-the-loop VLM，单位是交互 episode。
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md) — 专家并行仿真后端。
- [SMPC-to-RL](./paper-smpc2rl-loco-manipulation.md) — 另一条「仿真专家 → 真机策略」loco-manip 对照。
- [Manipulation](../tasks/manipulation.md) — 操作任务入口。
- [视频–接触–控制 10 篇技术地图](../overview/video-contact-control-10-papers-technology-map.md) — 本篇所在盘点坐标。

## 参考来源

- [Video2DoorTraversal 论文摘录](../../sources/papers/video2door_traversal_arxiv_2608_20251.md)
- [Video2DoorTraversal 项目页归档](../../sources/sites/video2door-traversal.md)
- [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)

## 推荐继续阅读

- [项目页](https://video2doortraversal.github.io/) — 管线视频、真机把手类型与 10/10 连续穿越
- [arXiv:2608.20251](https://arxiv.org/abs/2608.20251) — 方法公式、Table I–V 与消融
- [Articraft 项目页](https://articraft3d.github.io/) — DoorTwin 上游程序化关节资产
