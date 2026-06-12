# Multi-Agent Coordination for a Partially Observable and Dynamic Robot Soccer Environment with Limited Communication

> 来源归档（ingest · 人形机器人群控 / RoboCup SPL 有限通信）

- **标题：** Multi-Agent Coordination for a Partially Observable and Dynamic Robot Soccer Environment with Limited Communication
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2401.15026>
- **会议：** AIxIA 2023 / AIRO Workshop (CEUR-WS)
- **作者：** Daniele Affinita, Flavio Volpi, Valerio Spagnoli, Vincenzo Suriani, Daniele Nardi, Domenico D. Bloisi（Sapienza University of Rome 等）
- **平台：** RoboCup **Standard Platform League（SPL）** · **NAO 人形**
- **入库日期：** 2026-06-12
- **一句话说明：** 针对 SPL **WiFi 通信配额骤降**（2019–2022 队均包量 **−84%**；2023 上场 5→7 台且单包 128 B），提出 **分布式市场机制任务分配（DWM + DTA）**：各机本地融合队友模型、**Voronoi 站位修正**、无显式角色广播的自分配；真机 RoboCup 与 SimRobot 验证可 **显著减少任务重叠**。

## 核心论文摘录（MVP）

### 1) 通信约束背景（§1, Figure 1）

- **链接：** <https://arxiv.org/abs/2401.15026>
- **摘录要点：** RoboCup 2050 目标要求 **完全自主人形球队**；SPL 协调 **只能分布式、异步**。规则变化：队均允许包数从约 **5 pps/robot** 降至 **整场 1200 包/队**；2023 上场机器人 5→7 台，**每台可用包更少**；单包 **128 B**。四分之一决赛截图显示各队 **包计数器** 为硬约束。
- **对 wiki 的映射：**
  - [人形多机协调](../../wiki/concepts/humanoid-multi-robot-coordination.md) — 「有限通信」工程约束

### 2) 分布式世界模型 DWM（§3.1）

- **摘录要点：** 每机维护本地模型 $LM$（障碍/球/场地线）；接收队友 **事件** $e$（非常规率，如哨声、重新发现球）时用 $\delta$ 更新他人模型 $OLM_j$；**无事件时用预测模型**（障碍 GMM、球 Kalman、里程计）外推，避免频繁传全状态。合并得 **DWM** $= f(OLM_1,\ldots,LM)$。
- **对 wiki 的映射：**
  - [paper-humanoid-soccer-swarm-intelligence](../../wiki/entities/paper-humanoid-soccer-swarm-intelligence.md) — 对比：本文 **事件驱动** vs  swarm 论文 **周期性 UDP 状态广播**

### 3) 分布式任务分配 DTA（§3.2）

- **摘录要点：**
  - **Context Provider** 从 DWM 选战术上下文 CTX（优先级队列）。
  - **Voronoi 站位生成器** 产出 $N$ 个期望路点（角色无关）。
  - **效用矩阵 UEM**（$N\times M$）本地模拟拍卖；按上下文与路点过滤为 $N\times N$；$\Phi$ 最大化累计效用分配 **足球角色**（门将/前锋等）。
  - **关键：** 各机 **无需交换最终角色** 即可推断队友角色——同构算法 + 同 DWM 输入 → 一致分配。
- **对 wiki 的映射：**
  - [MARL](../../wiki/methods/marl.md) — 经典 **market-based task allocation** 在人形足球的落地

### 4) 实验与贡献（§1, §3）

- **摘录要点：** 在 **官方 RoboCup 比赛真机 NAO** 与 **SimRobot** 测试；低通信场景下 **任务重叠（task overlaps）显著减少**。方法统一了拍卖机制、角色映射与 Voronoi 站位，并强化 **长时间无网络更新** 时的自主性。
- **对 wiki 的映射：**
  - [Humanoid Soccer](../../wiki/tasks/humanoid-soccer.md)

## 对 wiki 的映射（汇总）

- [humanoid-multi-robot-coordination.md](../../wiki/concepts/humanoid-multi-robot-coordination.md)
- [paper-humanoid-soccer-swarm-intelligence.md](../../wiki/entities/paper-humanoid-soccer-swarm-intelligence.md) — SPL 有限通信对照

## 引用

```bibtex
@inproceedings{affinita2023multiagent,
  title     = {Multi-Agent Coordination for a Partially Observable and Dynamic
               Robot Soccer Environment with Limited Communication},
  author    = {Affinita, Daniele and Volpi, Flavio and Spagnoli, Valerio and
               Suriani, Vincenzo and Nardi, Daniele and Bloisi, Domenico D.},
  booktitle = {AIxIA 2023 -- AIRO Workshop},
  year      = {2023},
  eprint    = {2401.15026}
}
```
