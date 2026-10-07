---
type: entity
tags: [paper, humanoid, loco-manipulation, motion-retargeting, imitation-learning, data-generation, sim2real, uiuc, nvidia, unitree-g1, booster-k1, dexmate]
status: complete
updated: 2026-10-07
project_id: intermimicgen
project: https://sirui-xu.github.io/InterMimicGen/
arxiv: "2610.06850"
related:
  - ./paper-humanoidmimicgen.md
  - ../tasks/loco-manipulation.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ../overview/hub-wbt.md
sources:
  - ../../sources/papers/intermimicgen_arxiv_2610_06850.md
  - ../../sources/sites/intermimicgen.md
summary: "InterMimicGen（arXiv:2610.06850，UIUC/NVIDIA）将 InterAct 与 HiPHI 人体–物体交互重定向到 6 种机器人配置，以物理 PPO 跟踪器验证并自演进动作；五轮把验证集扩展到种子的约 142–151 倍，覆盖范围仍围绕已有任务语义。"
---

# InterMimicGen：用自演进动作模仿扩展人形移动操作

**InterMimicGen**（*Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation*，arXiv:2610.06850）由 **伊利诺伊大学厄巴纳-香槟分校（UIUC）** 与 **英伟达（NVIDIA）** 团队提出。它把人体–物体交互动作重定向为机器人参考，由物理跟踪策略执行并筛选，再用通过验证的变体共同扩展动作集与策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HOI | Human-Object Interaction | 人体与物体交互数据/动作 |
| PPO | Proximal Policy Optimization | 通用物理跟踪器的强化学习算法 |
| PD | Proportional-Derivative | 把关节目标转化为执行力矩的底层控制 |
| MPJPE | Mean Per-Joint Position Error | 关节位置平均误差 |
| UMR | Unified Motion Retargeting | 论文对照的全身运动重定向基线 |

## 核心信息

| 项 | 内容 |
|----|------|
| 论文 | arXiv 预印本，v1 提交于 2026-10-05 |
| 作者 | Yucheng Zhang、Sirui Xu、Jinhong Li、Liuyu Bian、Anatulya Nandi、Derek Zhang、Xiangchen Liu、Xueting Li、Umar Iqbal、Yu-Xiong Wang、Liang-Yan Gui |
| 机构 | 伊利诺伊大学厄巴纳-香槟分校（UIUC）；英伟达（NVIDIA） |
| 任务 | 从稀疏人体示范扩展可执行的人形全身移动操作动作 |
| 参考数据 | InterAct 与 HiPHI：16,059 段、140.68 小时、157 个对象条目 |
| 具身范围 | G1 普通手 / Inspire / Dex3、Booster T1 / K1、Dexmate Vega，共 6 种配置 |
| 项目与源码 | [官方项目页](https://sirui-xu.github.io/InterMimicGen/)提供论文和演示；截至 2026-10-07 未见此项目算法代码、训练数据或权重的官方发布入口 |

## 方法与核心原理

### 1. 整合人体–物体交互参考

InterAct 提供不同来源的日常交互类型，HiPHI 补充较长且同步的人体–物体运动。论文将其整理为共享身体、物体和手部表示；缺少手部标注的片段按“未知接触”处理。数据经目标平台筛选，排除物理仿真或机器人无法支持的片段。

### 2. 先全身保持关系，再细化手部接触

重定向分两级完成：

- **全身交互求解**：匹配人体标记点与物体表面的交互网格，保留长期身体–物体几何关系、支撑约束及关节限制；手部加入拇指与其他手指相向的抓取约束。
- **接触保持的几何清理**：两轮优化修正穿透、脚滑、局部抖动与手–物体漂移，同时限制轨迹偏离原动作。

这使接触语义和全身动作协同进入同一机器人参考，而非只拟合关节角或稀疏手腕点。

### 3. 用一个物理跟踪器执行多种交互

共享策略在并行仿真中用 PPO 训练，输入机器人本体状态、未来参考帧及身体链接到物体表面的几何关系，输出 PD 关节目标。奖励把身体运动、物体运动、身体–物体关系和手部接触组合起来；困难参考会被更频繁采样。

### 4. 让已验证的动作变成新一轮训练材料

每个候选从成功的父动作派生，编辑分为两类：

- **任务/物体编辑**：改变物体路径的位置或朝向，要求动作仍完成相同交互结果。
- **身体编辑**：改变下蹲深度、骨盆位置/朝向、站距、脚尖或肘部姿势，同时维持原有手部目标。

每轮将编辑候选与当前验证集一起用于跟踪器微调，再在物理仿真中回放。系统只接纳完成参考、未发生持续摔倒、动作平滑且任务结果/接触符合要求的轨迹，并从成功变体中选出下一轮父样本。这样策略能力与验证动作集同步扩展。

## 流程总览

{B}{B}{B}mermaid
flowchart TD
  data["InterAct + HiPHI<br/>人体-物体交互动作"]
  retarget["共享表示与平台筛选<br/>全身交互重定向"]
  hands["拇指对向约束<br/>接触保持几何清理"]
  tracker["PPO 物理跟踪器<br/>身体 + 物体 + 接触奖励"]
  edits["保持任务语义的小幅编辑<br/>物体路径 / 身体姿势"]
  sim["仿真执行与质量筛选"]
  archive["验证动作集 Dₖ₊₁"]
  finetune["跟踪器微调 πₖ₊₁"]
  robot["跨配置与真机演示"]
  data --> retarget --> hands --> tracker
  tracker --> edits --> finetune --> sim
  sim -->|通过的轨迹| archive
  archive --> edits
  archive --> finetune
  tracker --> robot
  sim --> robot
{B}{B}{B}

闭环里最关键的是“微调后再执行筛选”：通过物理验证的动作不仅增加训练数据，也把基策略原本无法跟踪的变化纳入后续能力范围。

## 评测与结果

- **数据规模：** InterAct + HiPHI 汇成 16,059 段动作、140.68 小时、157 个对象条目，并覆盖六种机器人配置。
- **接触保持重定向（G1 + Inspire，3,059 个 OMOMO 片段）：** 相对 Weave，帧级手接触保持率 **98.2% vs 96.1%**，物体穿透帧 **37.2% vs 67.4%**，脚滑帧 **12.3% vs 62.0%**，身体 MPJPE **5.74 cm vs 9.54 cm**。代价是手内滑移略高：**0.239 vs 0.225 m/s**。
- **单策略对多对象（15 个对象）：** G1 + Inspire 通用跟踪器在 bimanual / sitting / grasping 上成功率分别为 **73.2% / 64.9% / 58.3%**；每对象专用策略分别 **93.8% / 75.4% / 84.9%**。共享策略的身体误差较接近专用策略，差距主要在成功率与物体朝向控制。
- **五轮自演进：** 三种设置的验证动作量扩至种子的 **142.0–150.5 倍**。微调前，新生成动作成功率为 **52.1–64.2%**；微调后为 **98.4–98.9%**，原始动作在两者下均为 100%。新动作脚滑与关节加速度随演进轮次适度增加。
- **学习在飞轮中不可省：** 单独消融中冻结跟踪器第二轮的候选通过率降至 **0.4%**；微调跟踪器对应为 **72.9%**。只增加采样和筛选不能持续拓展动作范围。
- **真机演示：** G1 拉行李箱、G1 + Inspire 搬三脚架、Booster K1 抬放箱子、Dexmate Vega 移动椅子。测试控制栈因平台不同：G1 参考 50 Hz / 关节 PD 500 Hz，Dexmate 命令 100 Hz。

## 与其他工作对比

| 工作 | 数据生成与控制机制 | 本文中的边界 |
|------|--------------------|--------------|
| [HumanoidMimicGen](./paper-humanoidmimicgen.md) | 从 VR 示范拆分 object-centric 技能，结合全身规划与技能 DAG 生成轨迹 | 规划驱动的数据生成；InterMimicGen 从捕捉动作重定向出发，使用物理跟踪器反复微调与筛选 |
| MimicGen / DexMimicGen | 在条件变化下重放操作演示并筛选成功样本 | InterMimicGen 把该范式扩展到全身人形移动操作，并将策略学习纳入数据扩增闭环 |
| ULTRA | 通过场景缩放扩展人形交互参考 | InterMimicGen 另编辑全身姿势；论文报告身体运动覆盖范围更宽，但任务语义仍来自原演示 |
| Weave | 全身灵巧移动操作学习与重定向 | 重定向对照上 InterMimicGen 的多项接触/平滑指标更优，手内滑移略高 |

## 结论

**InterMimicGen 展示了一个由物理跟踪器驱动的人形交互数据飞轮：它把稀疏演示扩展成更多可执行变化，同时保留原有动作能力与任务结果。**

1. **更适合扩展已有交互，而非生成新任务。** 编辑保持交互类型、目标接触、物体和任务结果等约束，变化围绕种子演示。
2. **接触结构是重定向重点。** 全身交互网格与手指对向约束共同减少身体–物体关系和抓取的损失。
3. **通用策略承担数据生成角色。** 每轮策略微调是接纳更远动作变体的条件；冻结策略的后续候选通过率几乎归零。
4. **超过百倍的实验增长伴随可测代价。** 脚滑与关节加速度在后续轮次增加；评估数据集扩张时应同时看动作质量。
5. **共享能力仍未达到专用策略成功率。** 多对象通用化降低了成功率，抓取和物体旋转控制尤其困难。
6. **真机迁移已有多平台演示。** 论文按平台使用现有或训练得到的底层控制器；演示结果不等同于已发布完整可复现软件栈。

## 局限与风险

- 动作变体只能拓展已观测交互附近的覆盖，无法凭编辑创造新的任务语义。
- 迭代选择可能偏向较容易的对象，或让参考动作误差逐轮积累。
- 通用跟踪成功率落后于每对象专用策略，精细抓取和物体旋转是明显短板。
- 后期接受的动作离源示范更远，脚滑、身体关节加速度等质量指标有所变差。
- 实机效果依赖机器人平台、末端手型与其低层控制栈；不同平台的策略和动作支持并不完全相同。

## 工程实践

| 项 | 实践读法 |
|----|----------|
| 数据准备 | 需整合 InterAct / HiPHI，并按机器人模型、可用手型和对象仿真能力过滤动作 |
| 重定向 | 将身体–物体关系与接触保持纳入同一管线；分离全身求解和局部几何清理 |
| 训练 | 单策略便于扩展对象覆盖；对精细抓取任务，应与专用策略结果分别报告 |
| 动作扩增 | 保留父样本语义约束；微调、物理回放和质量筛选均是闭环必需步骤 |
| 真机验证 | 除任务完成外，监测脚滑、身体加速度、手部抖动及目标物体终态 |
| 官方实现状态 | 项目页截至 2026-10-07 未列出算法源码、训练数据或权重的发布链接；暂不能按公开仓库复现完整管线 |
| 源码运行时序图 | **不适用**：未找到由项目页链接的本项目可运行官方实现；论文提及的底层控制栈不构成该方法完整源码 |

## 关联页面

- [HumanoidMimicGen](./paper-humanoidmimicgen.md) — 规划驱动的人形移动操作示范生成
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 任务定义、全身移动与操作耦合
- [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md) — 参考重定向到物理跟踪的常见阶段
- [全身运动跟踪 WBT 知识链](../overview/hub-wbt.md) — 数据、跟踪训练与真机部署入口

## 参考来源

- [论文来源归档](../../sources/papers/intermimicgen_arxiv_2610_06850.md)
- [官方项目页归档](../../sources/sites/intermimicgen.md)
- [arXiv:2610.06850](https://arxiv.org/abs/2610.06850)
- [InterMimicGen 项目页](https://sirui-xu.github.io/InterMimicGen/)

## 推荐继续阅读

- [论文 HTML 全文](https://arxiv.org/html/2610.06850v1)
- [HumanoidMimicGen 项目与方法](./paper-humanoidmimicgen.md)
- [InterMimicGen 官方演示与图示](https://sirui-xu.github.io/InterMimicGen/)
