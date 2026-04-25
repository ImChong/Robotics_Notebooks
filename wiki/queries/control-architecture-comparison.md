---
type: query
tags: [wbc, rl, mpc, locomotion, humanoid, control]
status: complete
summary: "> **Query 产物**：本页由以下问题触发：「人形机器人的主流控制架构有哪些，各有什么优劣和适用场景？」"
updated: 2026-04-25
sources:
  - ../../sources/papers/whole_body_control.md
---

> **Query 产物**：本页由以下问题触发：「人形机器人的主流控制架构有哪些，各有什么优劣和适用场景？」
> 综合来源：[WBC vs RL](../comparisons/wbc-vs-rl.md)、[MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md)、[Reinforcement Learning](../methods/reinforcement-learning.md)、[Imitation Learning](../methods/imitation-learning.md)、[TSID](../concepts/tsid.md)

# 人形机器人控制架构综合对比

## 架构全景图

```
人形机器人控制架构
├── A. 经典控制路线
│   ├── A1. ZMP/LIPM + WBC（经典人形）
│   └── A2. MPC + WBC（当前工业主流）
├── B. 端到端 RL 路线
│   ├── B1. 纯 RL 端到端（输出关节位置）
│   └── B2. RL + 域随机化 + Sim2Real
├── C. 融合架构（当前学术前沿）
│   ├── C1. RL HLC + WBC LLC
│   ├── C2. AMP/ASE：RL + 运动风格学习
│   └── C3. 层次 IL：模仿学习分层
└── D. 统一架构（新兴）
    └── D1. 视觉-语言-动作（VLA）端到端
```

---

## A1. ZMP/LIPM + WBC（经典人形架构）

**代表**：ASIMO（Honda）、HRP 系列（AIST）、WABIAN（Waseda）

### 工作流程
```
步态规划（ZMP 条件）
  └→ 质心轨迹 + 落脚点序列
       └→ 逆运动学 / 逆动力学
            └→ 关节力矩
```

### 优劣
| | 说明 |
|-|------|
| ✅ 稳定性理论保证 | ZMP 在支撑多边形内 = 不摔倒 |
| ✅ 可解释、可调试 | 每个模块独立设计 |
| ❌ 运动缓慢保守 | ZMP 约束过于保守 |
| ❌ 无法应对非预期扰动 | 开环规划，动态响应差 |
| ❌ 需要精确接触模型 | 不平地形性能急剧下降 |

**适用**：研究用途、平地结构化环境、对稳定性有严格要求的场合。

---

## A2. MPC（Centroidal） + WBC（当前工业主流）

**代表**：ETH ANYmal（WBC 版）、MIT Cheetah 3、Boston Dynamics Atlas

### 工作流程
```
Centroidal MPC（1~50Hz）
  └→ 未来质心轨迹 + 接触力规划
       └→ WBC / TSID（200~1000Hz）
            └→ 关节加速度 / 力矩
```

### 优劣
| | 说明 |
|-|------|
| ✅ 动态稳定，可应对扰动 | MPC 在线优化，WBC 动力学一致 |
| ✅ 理论完善，产业成熟 | 大量工程验证（Boston Dynamics） |
| ✅ 安全约束可显式加入 | 关节限位、摩擦锥等硬约束 |
| ❌ 依赖精确模型 | SysID 工作量大 |
| ❌ 工程复杂度高 | 多层模块调参困难 |
| ❌ 接触切换仍需状态机 | 非接触瞬间容易失稳 |

**适用**：工业级人形机器人、需要精确任务执行、有充足调参工程资源。

---

## B1/B2. 端到端 RL

**代表**：ETH Learning to Walk in Minutes、Unitree legged_gym、OpenAI Dactyl

### 工作流程
```
PPO/SAC 在仿真中大规模训练（域随机化）
  └→ 神经网络策略 π(a|s)
       └→ 直接输出关节位置目标
            └→ PD 控制器执行
```

### 优劣
| | 说明 |
|-|------|
| ✅ 实现简单 | 无需手工设计控制层次 |
| ✅ 接触切换自然 | 端到端学习处理所有情况 |
| ✅ 强泛化性 | 域随机化后可跨地形迁移 |
| ❌ 可解释性差 | 失败原因难定位 |
| ❌ Reward 设计难 | 需要大量 reward engineering |
| ❌ Sim2Real gap | 真实机器人部署仍需大量工作 |
| ❌ 行为可能不自然 | 策略优化 reward，不保证运动美观 |

**适用**：研究快速原型、复杂地形 locomotion、不需要精确轨迹跟踪。

---

## C1. RL HLC + WBC LLC（融合主流）

**代表**：ETH ANYmal（RL 版）、Agility Cassie、多数 2022 年后顶会工作

### 工作流程
```
RL High-Level Controller（50~100Hz）
  └→ 质心速度指令 / 落脚点
       └→ WBC Low-Level Controller（200~1000Hz）
            └→ 关节力矩
```

### 优劣
| | 说明 |
|-|------|
| ✅ 两层解耦，各自优化 | RL 处理策略，WBC 保证物理一致 |
| ✅ 安全约束保留 | WBC 层可加入硬约束 |
| ✅ 训练更高效 | RL 只需学高层策略，动作空间降维 |
| ❌ 两层接口设计复杂 | 高层输出格式和低层输入需仔细对齐 |
| ❌ 仍依赖 WBC 模型 | 模型不准则低层执行失效 |

**适用**：需要安全约束 + 高泛化性的工业部署场景。

---

## C2. AMP/ASE：RL + 运动风格学习

**代表**：AMP（Peng et al. 2021）、ASE（Peng et al. 2022）、PHC（ICCV 2023）

### 工作流程
```
MoCap 数据 → 训练判别器 D（区分真实/生成动作）
  └→ 判别器 reward + 任务 reward → PPO 训练 Actor
       └→ 自然步态 + 任务完成
```

**ASE 扩展**：
```
预训练阶段：LLC 学习多样化运动技能嵌入 z
推理阶段：HLC 在潜空间 Z 上选择技能，LLC 执行
```

### 优劣
| | 说明 |
|-|------|
| ✅ 自然运动风格 | MoCap 数据驱动，无需手工设计动作 |
| ✅ 技能复用 | ASE 的技能嵌入可跨任务 |
| ❌ 需要 MoCap 数据 | 数据采集和重定向成本较高 |
| ❌ 判别器不稳定 | 类似 GAN 的训练挑战 |

**适用**：需要自然人类化步态、服务机器人、影视/游戏机器人。

---

## C3. 层次模仿学习

**代表**：ACT（Action Chunking Transformers）、Diffusion Policy 层次版

### 工作流程
```
高层策略：语言/视觉 → 任务分解（子目标序列）
  └→ 低层策略：子目标 → 关节动作序列（BC/扩散）
```

### 优劣
| | 说明 |
|-|------|
| ✅ 适合操作任务 | 人类演示直接转化为策略 |
| ✅ 数据效率高 | 不需要大量 RL 训练 |
| ❌ 受演示质量限制 | 超不过专家上界 |
| ❌ 分布外泛化差 | 新场景需要新演示 |

**适用**：精细操作（装配、服务）、有大量人类演示数据。

---

## D1. VLA（视觉-语言-动作统一模型）

**代表**：RT-2（Google）、OpenVLA、π0（Physical Intelligence）

### 工作流程
```
语言指令 + 视觉输入 → 大型多模态模型 → 关节动作
```

### 优劣
| | 说明 |
|-|------|
| ✅ 零样本泛化 | 语言驱动，可以执行新指令 |
| ✅ 多任务统一 | 一个模型处理所有任务 |
| ❌ 计算开销极大 | 推理需要高端 GPU |
| ❌ 控制频率低 | 大模型难以 >10Hz |
| ❌ 成熟度低 | 仍是研究阶段 |

**适用**：通用机器人、研究前沿、不需要高频精确控制。

---

## 选型决策树

```
你的场景是什么？
│
├── 需要自然运动 + 精确安全约束
│   └→ AMP/ASE + WBC（C2 + A2 融合）
│
├── 复杂地形 locomotion（快速部署）
│   └→ 端到端 RL + 域随机化（B2）
│
├── 精细操作（灵巧手/装配）
│   └→ 层次 IL（C3）
│
├── 工业级产品（可靠性优先）
│   └→ MPC + WBC（A2）
│
├── 研究快速原型
│   └→ legged_gym + PPO（B1）
│
└── 通用机器人（未来方向）
    └→ VLA（D1）
```

---

## 关联页面

- [WBC vs RL](../comparisons/wbc-vs-rl.md) — 两条路线详细对比和融合架构
- [MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md) — A2 架构详解
- [TSID](../concepts/tsid.md) — WBC 低层执行器
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 方法全景
- [Imitation Learning](../methods/imitation-learning.md) — IL 方法全景
- [Diffusion Policy](../methods/diffusion-policy.md) — 层次 IL 的现代实现
- [RL vs IL](../comparisons/rl-vs-il.md) — B vs C 路线对比

## 参考来源

- Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control* (2021)
- Peng et al., *ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters* (2022)
- Sentis & Khatib, *Synthesis of Whole-Body Behaviors through Contact and Collision Avoidance* — WBC 理论基础
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (2023) — VLA 架构
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md) — TSID/HQP/Crocoddyl ingest 摘要
