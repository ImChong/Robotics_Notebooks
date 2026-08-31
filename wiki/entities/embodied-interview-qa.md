---
type: entity
tags: [interview, embodied-ai, vla, reinforcement-learning, imitation-learning, sim2real, education, open-source]
status: complete
updated: 2026-08-08
related:
  - ./coding-interview-university.md
  - ./lumina-embodied.md
  - ./learn-robotics-qqfly-guide.md
  - ./waytoagi.md
  - ./humanoid-system-curriculum.md
  - ../methods/vla.md
  - ../methods/reinforcement-learning.md
  - ../methods/imitation-learning.md
  - ../methods/ppo.md
  - ../methods/diffusion-policy.md
  - ../concepts/sim2real.md
  - ../concepts/whole-body-control.md
  - ../concepts/deep-learning-foundations.md
  - ../concepts/domain-randomization.md
  - ../tasks/humanoid-locomotion.md
  - ../tasks/vision-language-navigation.md
  - ../overview/vla-open-source-repro-landscape-2025.md
  - ../../roadmap/motion-control.md
  - ../../roadmap/depth-vla.md
  - ../../roadmap/depth-rl-locomotion.md
  - ../../roadmap/depth-sim2real.md
sources:
  - ../../sources/sites/embodied-interview-qa-github-io.md
  - ../../sources/repos/embodied-interview-qa.md
summary: "WinstonJQ 开源中文具身智能高频面试题库（MIT / GitHub Pages）：八卷折叠问答覆盖通识、RL、VLA·IL、世界模型·Sim2Real、工程落地、腿足控制、感知导航与 LeetCode·系统设计；按频次与 L1–L3 分层，与本库方法页互补做面试前补盲。"
---

# 具身智能高频面试题库（Embodied Interview QA）

**一句话：** [winstonjq.github.io/embodied-interview-qa](https://winstonjq.github.io/embodied-interview-qa/) 是 WinstonJQ 维护的 **MIT 中文面试题库**：从公开面经合并高频题，按八卷主题 + L1/L2/L3 + 频次标签组织短答案；适合面试前 30 分钟补盲区，**不替代**本库对机制、开源状态与路线图的结构化编译。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 卷三主轴：视觉–语言–动作模型与动作生成 |
| IL | Imitation Learning | 行为克隆 / DAgger / ACT / Diffusion Policy 等 |
| RL | Reinforcement Learning | 卷二与卷六：算法本身 vs 腿足落地 |
| Sim2Real | Simulation to Reality | 卷四：域随机化、蒸馏与仿真栈选型 |
| WBC | Whole-Body Control | 卷六：全身控制 / TSID 与 MPC 栈 |
| VLN | Vision-Language Navigation | 卷七：语言条件下导航与 ObjectNav |
| PPO | Proximal Policy Optimization | 卷二高频；人形 locomotion 与 RLHF 常用 |
| ACT | Action Chunking with Transformers | 卷三动作块模仿学习代表 |

## 为什么重要

1. **填「面试速查」层**：本库强在机制与复现核查；本题库强在**频次信号 + 短答案 + 易错一句**，两者正交。
2. **八卷覆盖本库主线**：VLA / RL / Sim2Real / 腿足 WBC / VLN 与 [motion-control](../../roadmap/motion-control.md)、[depth-vla](../../roadmap/depth-vla.md)、[depth-rl-locomotion](../../roadmap/depth-rl-locomotion.md)、[depth-sim2real](../../roadmap/depth-sim2real.md) 可直接对照刷。
3. **开源可维护**：源码与 Pages 同源（MIT）；贡献要求题源来自公开面经，避免编造题污染可信度。

## 流程总览（推荐用法）

```mermaid
flowchart LR
  GOAL["岗位方向\nVLA / RL / 腿足 / 导航"]
  VOL["选卷\nL1→L2→L3"]
  SELF["折叠自测\n先答再展开"]
  WIKI["跳回本库\n方法/概念深读"]
  ROAD["对照 roadmap\n补系统缺口"]

  GOAL --> VOL --> SELF --> WIKI --> ROAD
```

| 阶段 | 做法 | 本库用法 |
|------|------|----------|
| 选卷 | 按 JD 选 VOL 03/02/06/07 等 | 对照下表「卷 ↔ wiki」 |
| 自测 | 默认折叠，先口述再点开 | 易错句当 checklist |
| 深读 | 答不出或答案过浅时 | 进方法页 / 论文实体，勿停在题库 |
| 系统补齐 | 多卷盲区成片 | 回 [运动控制主路线](../../roadmap/motion-control.md) |

## 八卷 ↔ 本库节点映射

| 卷 | 主题（宣称题数） | 优先对照本库 |
|----|------------------|--------------|
| 01 | 通识基础（55） | [深度学习基础](../concepts/deep-learning-foundations.md)、[RL](../methods/reinforcement-learning.md)、[Transformer](../concepts/transformer.md) |
| 02 | RL 算法（50） | [RL](../methods/reinforcement-learning.md)、[PPO](../methods/ppo.md)、[RL vs IL](../comparisons/rl-vs-il.md) |
| 03 | VLA / 模仿学习（77） | [VLA](../methods/vla.md)、[Imitation Learning](../methods/imitation-learning.md)、[Diffusion Policy](../methods/diffusion-policy.md)、[Action Chunking](../methods/action-chunking.md)、[VLA 复现地图](../overview/vla-open-source-repro-landscape-2025.md) |
| 04 | 世界模型 / Sim2Real（31） | [Sim2Real](../concepts/sim2real.md)、[Domain Randomization](../concepts/domain-randomization.md)、[World–Action Models](../concepts/world-action-models.md)、[Isaac Lab](./isaac-lab.md) |
| 05 | 工程落地（47） | [VLA 部署 query](../queries/vla-deployment-guide.md)、遥操作 / 数据飞轮相关实体、分布式训练选型笔记 |
| 06 | 腿足控制 / 遥操作（58） | [Whole-Body Control](../concepts/whole-body-control.md)、[Humanoid Locomotion](../tasks/humanoid-locomotion.md)、[Actuator Network](../methods/actuator-network.md)、[depth-rl-locomotion](../../roadmap/depth-rl-locomotion.md) |
| 07 | 3D 感知 / SLAM / VLN（67） | [VLN](../tasks/vision-language-navigation.md)、[导航·SLAM 栈](../overview/navigation-slam-autonomy-stack.md)、[depth-navigation](../../roadmap/depth-navigation.md) |
| 08 | LeetCode + 系统设计（40） | 通用 coding；系统设计题对照 VLA 服务 / 感知–规划–控制全栈相关 overview |

README 宣称主表及补充约 **425** 题；入库日 Markdown `<summary>` 合计约 **438**（含手撕与补充，以仓为准）。

## 核心结构 / 机制

**筛选规则（项目自述）：** 公开面经同义合并 → 频次 ≥3 入主表；近期集中但未满三源标「补充」。不分公司标签，按技术主题成卷。

**答案形态：** ≤350 字精简答 +「易错」；超长拆「答 / 关键对比 / 易错」。部分卷含 **§H 手撕**（公式 / 短 Python）；卷八 LeetCode 段给考察点与短实现，不贴长篇题解。

**交互：** HTML5 `<details>` 零 JS 折叠——天然支持「先自测再对答案」。

**生成与审查：** 维护者披露用 multi-agent 起草 + 跨模型二次审查；读者仍应以论文与本库实体核对关键事实，题库是**速记层**不是权威层。

## 工程实践

| 场景 | 建议 |
|------|------|
| **VLA 岗二面** | 先刷卷三 L1→L2，再对照 [VLA](../methods/vla.md) / [π₀](../methods/π0-policy.md) / [OpenVLA](./openvla.md) |
| **人形 / 四足运控** | 卷六 + 卷二（算法）交叉；盲区回 [WBC](../concepts/whole-body-control.md) 与 [RL locomotion 纵深](../../roadmap/depth-rl-locomotion.md) |
| **Sim2Real 专项** | 卷四；部署坑见 [Sim2Real](../concepts/sim2real.md) 与 [闭环误差分层](../queries/sim2real-closed-loop-engineering.md) |
| **贡献新题** | 仓 README 格式；必须附公开面经来源，勿编造 |

### 开源状态（步骤 2.5）

- **已开源（MIT）**：站点 + 题库 Markdown + 渲染工具 — [GitHub](https://github.com/WinstonJQ/embodied-interview-qa)
- **不适用训练/推理时序图**：本题库不是可运行算法实现；无 `train.py` / 权重发布义务

## 局限与风险

- **误区：刷完题库 = 会做研究/上真机。** 短答案刻意省略推导与完整代码；工程细节以本库与一手论文为准。
- **误区：AI 二次审查 = 零事实错误。** 仍可能过时或简化过度（尤其前沿 VLA 数字）；关键指标回实体页。
- **误区：频次标签是公司 JD 保证。** 频次来自公开面经合并统计，不是某司必考清单。
- **局限：** 卷八与各卷手撕偏通识 coding；与具身专识卷正交，勿用其替代运控深读。
- **与其它中文入口分工：** [Lumina Guide](./lumina-embodied.md) 偏百科/社区；[qqfly 指南](./learn-robotics-qqfly-guide.md) 偏机械臂规控自学；[WaytoAGI](./waytoagi.md) 偏大众 AI 雷达；[Coding Interview University](./coding-interview-university.md) 偏 **通用 SWE 算法/刷题路线图**；**本题库偏面试速查**。

## 关联页面

- [Coding Interview University](./coding-interview-university.md) — 大厂 SWE 通用 CS/刷题路线图（补卷八 coding 底座）
- [Lumina 具身智能社区](./lumina-embodied.md) — 百科 Guide / Talks，与面试题库互补
- [开源机器人学学习指南（qqfly）](./learn-robotics-qqfly-guide.md) — 系统自学手册（固定基臂主线）
- [WaytoAGI](./waytoagi.md) — 中文社区飞书雷达
- [人形系统学习策展](./humanoid-system-curriculum.md) — 课程章映射，非面试速查
- [VLA](../methods/vla.md) · [Imitation Learning](../methods/imitation-learning.md) · [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Sim2Real](../concepts/sim2real.md) · [Whole-Body Control](../concepts/whole-body-control.md) · [深度学习基础](../concepts/deep-learning-foundations.md)
- [VLA 开源复现地图（2025）](../overview/vla-open-source-repro-landscape-2025.md)
- [运动控制主路线](../../roadmap/motion-control.md) · [VLA 纵深](../../roadmap/depth-vla.md) · [RL 纵深](../../roadmap/depth-rl-locomotion.md) · [Sim2Real 纵深](../../roadmap/depth-sim2real.md)

## 参考来源

- [sources/sites/embodied-interview-qa-github-io.md](../../sources/sites/embodied-interview-qa-github-io.md) — 项目页归档与开源核查
- [sources/repos/embodied-interview-qa.md](../../sources/repos/embodied-interview-qa.md) — GitHub 仓结构与贡献入口
- 在线题库：<https://winstonjq.github.io/embodied-interview-qa/>
- 源码：<https://github.com/WinstonJQ/embodied-interview-qa>

## 推荐继续阅读

- 卷三在线页（VLA/IL）：<https://winstonjq.github.io/embodied-interview-qa/interviews/03_vla_il.html>
- 卷六在线页（腿足/遥操作）：<https://winstonjq.github.io/embodied-interview-qa/interviews/06_legged_control.html>
- [Embodied-AI-Guide（Lumina）](https://github.com/tianxingchen/Embodied-AI-Guide) — 百科全书式对照阅读
- [Xbotics-Embodied-Guide](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide) — 工程路线图对照
