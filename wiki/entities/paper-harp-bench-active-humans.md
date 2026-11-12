---
type: entity
project_id: harp-bench-active-humans
title: HARP-Bench（arXiv:2611.11123）
tags:
  - paper
  - benchmark
  - human-robot-interaction
  - manipulation
  - vla
  - meta
status: complete
updated: 2026-11-11
arxiv: "2611.11123"
code: https://github.com/UMass-Embodied-AGI/harp
original:
  - ../../sources/papers/harp-bench_arxiv_2611_11123.md
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/action-chunking.md
  - ../methods/diffusion-policy.md
  - ../entities/paper-pi0.md
  - ../entities/paper-smolvla.md
  - ../entities/lerobot.md
  - ../entities/genesis-sim.md
  - ../entities/robotwin.md
  - ../entities/robocasa.md
  - ../entities/libero-benchmark.md
  - ../concepts/sim2real.md
  - ../concepts/safety-filter.md
sources:
  - ../../sources/papers/harp-bench_arxiv_2611_11123.md
summary: "HARP-Bench（arXiv:2611.11123，CoRL 2026 稿）：Genesis 仿真基准，37 个任务按 Cueing / Contribution / Interruption / Co-presence 四类主动人类行为分组，闭环评测任务成功率与碰撞 / 距离指标，附脚本示教生成与 LeRobot 兼容接口；最佳学习策略 π0 仅 0.19，真机仅在 xArm dump-bin 单任务族上验证。"
---

# HARP-Bench（arXiv:2611.11123）

**HARP-Bench**（*HARP-Bench: Benchmarking Robot Manipulation Around Active Humans*，[arXiv:2611.11123](https://arxiv.org/abs/2611.11123)，马萨诸塞大学阿默斯特分校、明尼苏达大学、Meta 等，CoRL 2026 稿）。原文归档见 [sources/papers/harp-bench_arxiv_2611_11123.md](../../sources/papers/harp-bench_arxiv_2611_11123.md)，代码 [UMass-Embodied-AGI/harp](https://github.com/UMass-Embodied-AGI/harp)，项目页 <https://harp-bench.vercel.app/>。

## 一句话定义

**一个仿真操作基准：把「在工作空间里主动行动的人」当作操作问题的一部分，用 37 个任务 + 脚本化动画头像，测策略在人给提示、帮忙、打断、或只是在旁边干活时的成功率与安全指标，并附示教生成管线（论文摘要、§3）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HARP | 论文未展开缩写；标题为 Robot Manipulation Around Active Humans | 基准名，「主动人类周围的机器人操作」（推测其含义，论文未给全称） |
| VLA | Vision-Language-Action | 视觉-语言-动作策略，本文测 π0、SmolVLA |
| ACT | Action Chunking with Transformers | 动作分块模仿学习基线（§4.1） |
| DP | Diffusion Policy | 扩散式模仿学习基线（§4.1） |
| HRI / HRC | Human-Robot Interaction / Collaboration | 人机交互 / 协作；论文把四类条件对应到 HRI 经典问题（§3.1） |
| AMASS | Archive of Motion Capture as Surface Shapes | 头像动作来源之一（§3.2） |
| RGB-D | Red-Green-Blue + Depth | 基准提供的观测形式（§1） |
| IK / RRT | Inverse Kinematics / Rapidly-exploring Random Tree | 脚本专家使用的求解与规划（§4.1） |

## 为什么重要

- 现有操作基准大多只变化物体、布局、指令等「被动因素」，机器人是唯一改变工作空间的主体（作者自述，§1）；HARP-Bench 补上「人在场且在动」这一维。
- 它同时报告成功率与碰撞 / 距离类安全指标，并指出成功率高不一定更安全（§4.2、表 3），对评测协议有参考价值。
- 成对的「有人 / 无人」变体让「人的影响」可以单独量化（§3.1、表 4）。

## 核心原理与流程总览

**四类条件（§3.1、图 1）：** Cueing（人的动作暗示目标）、Contribution（人完成一部分子任务）、Interruption（人检查 / 占用物体，改变可用性或时机）、Co-presence（人在旁边做无关的事，增加运动约束）。作者说明这是隔离对操作的影响，不是人类行为的完备分类。

```mermaid
flowchart LR
  A[任务初始化 物体 机器人 指令] --> B[事件调度器触发头像动作]
  B --> C[闭环仿真 策略读 RGB-D 与状态出动作]
  C --> D[指标 成功 碰撞 最小距离 近距占比]
  A --> E[同一设置下脚本专家 rollout]
  E --> F[数据记录器 LeRobot 兼容接口]
  F --> G[训练 ACT DP pi0 SmolVLA]
  G --> C
```

要点（均出自论文）：

- **头像与事件（§3.2、附录 C.2）：** 46 个蒙皮头像外观继承自 Virtual Community；动作来自 Mixamo 与 AMASS，人工裁剪并标注物体交互关键帧，可做手物附着，部分场景用 IK 调整。人侧事件是脚本化的，机器人侧是物理仿真。触发：Contribution 在早期随机时刻开始，Interruption 依赖夹爪位置，Co-presence 随机选无关动作。
- **指标（§3.2）：** 成功要求完成且无机器人 - 头像碰撞；另报 Collision、Min. Dist.、Violat.（距离小于 0.10 m 的时间步比例）。
## 评测设置与基线结果

- **评测设置（§4.1）：** 训练 11,100 条脚本示教（37 × 300），每任务 100 个评测种子。两个专家参考：全状态专家（能看到头像位姿与事件子目标）与无意图专家。
- **关键结果（§4.2）：**

| 策略 | 成功率（表 2） | Collision（表 3） | Min. Dist.（表 3） |
|------|------|------|------|
| 全状态专家 | 0.99 | 0.000 | 0.158 m |
| 无意图专家 | 0.48 | 0.299 | 0.126 m |
| π0 | 0.19 | 0.207 | 0.095 m |
| ACT | 0.10 | 0.102 | 0.133 m |
| DP | 0.09 | 0.174 | 0.120 m |
| SmolVLA | 0.06 | 0.214 | 0.104 m |

按条件看（表 2 平均行）：π0 在 Co-presence 0.31、Cueing 0.16、Contribution 0.16、Interruption 0.09；无意图专家在 Interruption 为 0.00，在 Co-presence 为 0.97。

补充结果：逐任务训练的 ACT 加入人类变体示教后，25 任务子集的人类变体平均成功率 0.10→0.15，基础任务 0.42→0.41（表 4）；学习策略失败中「操作 / 恢复」占 60–66%、碰撞占 22–27%（图 3）；真机 xArm dump bin 任务族上 Contribution 0.13→0.33，Interruption 仍为 0.00（表 5）。

## 与其他工作对比

| 工作 | 关注点 | 与本文的区别 |
|------|--------|--------------|
| [LIBERO](./libero-benchmark.md) | 终身学习操作基准，变化物体、布局与指令 | 工作空间里只有机器人在改变环境；HARP-Bench 把会动的人作为变量，并加入碰撞 / 距离安全指标 |
| [RoboCasa](./robocasa.md) | 大规模厨房日常任务 MuJoCo 仿真基准 | 场景与任务规模远大，但无主动人类；HARP-Bench 只有 37 个桌面任务 - 条件实例 |
| [RoboTwin](./robotwin.md) | 双臂操作的自动数据生成与仿真平台 | 同样提供脚本专家生成示教；HARP-Bench 的示教与评测都围绕人类事件触发展开 |

## 结论

**HARP-Bench 表明「人在场且在动」是现有操作策略的明显短板：脚本专家可达 0.99，而最佳学习策略 π0 只有 0.19。**

1. 四类条件（Cueing、Contribution、Interruption、Co-presence）把人的影响拆开评测，成对的有人 / 无人变体可单独量化人的作用（§3.1）。
2. 成功率与安全指标并不一致：π0 成功率最高但碰撞与最小距离更差，ACT 安全指标最好（表 3）。
3. Interruption 最难，无意图专家与真机都为 0.00，说明需要理解人的意图而非只做避障（表 2、表 5）。
4. 局限：人是脚本化头像、无持续适应；以桌面操作为主；10 cm 阈值只是评测约定；真机只覆盖一个任务族，仿真安全不等于部署安全（§6）。

## 工程实践

- **运行门槛（仓库 README，静态阅读，未运行）：** Python 3.10、NVIDIA GPU 且需 Vulkan 才能用 NYX 渲染（README 称 A100 / H100 不能跑 NYX）；至少 32 GB 内存；依赖 `genesis-world==1.2.0`（见 [Genesis 仿真器](../entities/genesis-sim.md)）。
- **资源：** 资产包与基线检查点公开在 Hugging Face（README）；README 称 LeRobot 训练数据集不提供下载，需用脚本专家自行采集。
- **数据接口：** 与 [LeRobot](../entities/lerobot.md) 兼容，另有转换路径（§3.3）。
- **渲染开销（附录 A、表 A1）：** A16 GPU 640×480：无头像 203.01 FPS，动画头像 89.55 FPS；论文称 20 Hz 采集下约 4.5 倍实时。
- **真机复现的最小路径（§4.3）：** xArm 桌面，dump bin 任务族，ACT，仅 Contribution 与 Interruption 两个变体，每设置 30 次。

## 与文章说法的核对

文章（2026-11-11《机器人动作块太长，什么时候该重新规划？》的 53 篇速览之第 07 条）写道：把人的提示、协助、打断与共处加入操作环境；基准以仿真人物为主，配有有限真机验证；任务成功率与人机交互安全仍需分别报告。

| 文章说法 | 核对结果 |
|------|------|
| 提示、协助、打断与共处 | 对应 Cueing、Contribution、Interruption、Co-presence（§3.1）。「协助」略偏：Contribution 指人完成一部分子任务，机器人做剩余部分，不是人去协助机器人的所有情形。 |
| 以仿真人物为主 | 属实：基准是仿真，人是脚本化动画头像（§3.2），论文也承认不涵盖真实人类行为的完整变化。 |
| 有限真机验证 | 属实且需更具体：仅 xArm、单个 dump bin 任务族、仅 ACT、Contribution 与 Interruption 两个变体（§4.3、表 5）；Contribution 0.13 到 0.33，Interruption 仍为 0.00。作者明说未验证全部四类条件或真实安全。文章未给这些数字。 |
| 成功率与安全需分别报告 | 与表 3 一致：π0 成功率最高（0.19），ACT 三项安全指标最好（§4.2）。作者另指出碰撞本身已计为失败，故不是两个独立量；文章「分别报告」的说法可读为指标不可互相替代。 |
| 文章未提的重要限定 | 学习策略整体成功率很低（最好 0.19）；10 cm 阈值是评测约定而非 ISO/TS 15066 规定的防护距离（§3.2）；仿真安全不等于部署安全（§6）。 |

未发现文章与论文在数字上的冲突（文章该段不含数字）。

## 局限

**论文自述（§3.2、§4.3、§6）：**

- 脚本化头像不体现持续的人机适应与人类意图、动作、沟通的完整变化；自适应人类模型留作未来工作。
- 以桌面操作为主，交互多样性有限，无难度递进；不含移动操作与长程协作。
- 感知与接触动力学简化；成功率与几何安全指标不直接衡量等待、绕行、识别人完成的子任务或社交舒适度。
- 37 个是任务 - 条件实例，不是 37 种不同技能（§3.1）；评测种子变化不度量不同训练运行之间的方差（§4.1）。
- 真机只覆盖一个任务族且 Interruption 无提升，Co-presence 因无法精确测人机间距未做（§4.3）。

**阅读时看到的未验证之处（作者判断以外，属我的观察）：**

- 学习策略成功率集中在 0.0 到 0.3 区间，大量任务为 0.00（表 2），区分度可能受「基线本身很弱」影响；这些是否来自示教数据量、策略结构或任务难度，论文未做消融（推测）。
- 表 4 只用 ACT 逐任务训练，未对 VLA 做同样的人类变体数据对照。
- 失败分类里「意图」类占比正文未给数；「操作 / 恢复」类含交互引发的错误，不等同基础操作失败（作者在 §4.2 亦有提示）。
- 论文正文未交代仿真机器人型号；README 与论文的评测种子数口径不同（见原文归档），未核实原因。
- 本页未运行任何代码，未复现表中数字。

## 关联页面

- [VLA 方法页](../methods/vla.md)
- [模仿学习](../methods/imitation-learning.md)
- [动作分块 Action Chunking](../methods/action-chunking.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [π0 论文页](../entities/paper-pi0.md)
- [SmolVLA 论文页](../entities/paper-smolvla.md)
- [LeRobot](../entities/lerobot.md)
- [Genesis 仿真器](../entities/genesis-sim.md)
- [RoboTwin](../entities/robotwin.md)、[RoboCasa](../entities/robocasa.md)、[LIBERO](../entities/libero-benchmark.md)（论文表 1 对比的同类操作基准，LIBERO 在引言中被引）
- [Sim2Real](../concepts/sim2real.md)
- [安全滤波器 Safety Filter](../concepts/safety-filter.md)
- [具身评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)：选基准时可把「有无主动人类」作为一个维度，HARP-Bench 是该维度的代表

## 参考来源

- [sources/papers/harp-bench_arxiv_2611_11123.md](../../sources/papers/harp-bench_arxiv_2611_11123.md)
- [arXiv:2611.11123](https://arxiv.org/abs/2611.11123)
- 代码：<https://github.com/UMass-Embodied-AGI/harp>；项目页：<https://harp-bench.vercel.app/>

## 继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.11123v1)
- [VLA 方法页](../methods/vla.md)
- [Sim2Real](../concepts/sim2real.md)
