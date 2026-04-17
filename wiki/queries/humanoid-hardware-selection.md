---
title: 人形机器人硬件选型指南
type: query
status: complete
created: 2026-04-14
updated: 2026-04-14
summary: 对比当前主流人形机器人平台（G1 / H1 / Unitree B2 / Figure / Atlas），从研究场景和工程目标给出选型建议。
sources:
  - ../../sources/papers/humanoid_hardware.md
---

> **Query 产物**：本页由以下问题触发：「做人形机器人运动控制研究，该选哪个硬件平台？」
> 综合来源：[Locomotion](../tasks/locomotion.md)、[Sim2Real](../concepts/sim2real.md)、[Loco-Manipulation](../tasks/loco-manipulation.md)

# 人形机器人硬件选型指南

## 核心选型维度

在比较具体平台之前，先确定你的主要约束：

| 约束 | 推荐方向 |
|------|---------|
| 学术研究预算（< 200K CNY）| Unitree H1 / G1 |
| 商业应用，需要技术支持 | Agility Cassie / Figure |
| 复杂操作（双手+locomotion）| Unitree G1 / Fourier GR1 |
| 极限性能测试 | Boston Dynamics Atlas（不对外销售） |
| 仿真先行，暂不购买硬件 | MuJoCo 官方模型 + Isaac Lab |

---

## 主流平台对比

| 平台 | 开发商 | DoF | 身高 | 重量 | 价格范围 | 开源程度 |
|------|--------|-----|------|------|---------|---------|
| **Unitree G1** | Unitree | 43 | 1.27m | 35kg | ~90K CNY | 较高（SDK+URDF） |
| **Unitree H1** | Unitree | 19 | 1.8m | 47kg | ~90K CNY | 较高（SDK+URDF）|
| **Fourier GR1** | Fourier Intelligence | 44 | 1.65m | 55kg | ~150K CNY | 中等 |
| **UBTECH Walker X** | 优必选 | 41 | 1.7m | 76kg | 未公开 | 低 |
| **Figure 02** | Figure AI | ~50 | 1.7m | 60kg | 未公开 | 低 |
| **Atlas (BD)** | Boston Dynamics | 28 | 1.5m | 89kg | 不对外销 | 无 |

---

## 各平台详细分析

### Unitree G1
**适合：** 学术研究、全身操作、loco-manipulation

**优势：**
- 43 自由度（含手指），可做精细操作
- 价格相对低，学术界可承受
- 良好的开源生态（legged_gym + IsaacLab G1 模型）
- Unitree SDK2 支持 Python/C++ 控制

**劣势：**
- 较矮（1.27m），部分场景（开门、货架操作）受限
- 关节力矩相对较小（非 SEA 驱动）

**典型用途：** loco-manipulation 研究、RL 全身控制、中文学术圈主流

---

### Unitree H1
**适合：** locomotion 研究、高速移动、赛跑任务

**优势：**
- 更高（1.8m）更重，locomotion 更稳定
- 关节力矩大，适合动态运动
- 同样良好的 SDK 支持

**劣势：**
- DoF 较少（无手，操作能力有限）
- 速度场景无双臂

**典型用途：** 双足 locomotion RL、地形穿越、户外测试

---

### Fourier GR1
**适合：** 工业/服务场景验证、操作研究

**优势：**
- 成熟的工业化设计，可靠性高
- 全身 44 DoF，含灵巧手选项
- 国内工业界有实际落地

**劣势：**
- 价格较高
- 开源生态不如 Unitree 丰富

---

### Figure 02 / Agility Digit
**适合：** 仓储 / 物流场景，海外合作

**特点：**
- 面向商业落地，不是学术平台
- Agility Digit 有较成熟的 locomotion 控制系统（来自 Cassie 积累）
- Figure 背后有 OpenAI 合作，VLA / 具身智能路线

---

## 仿真平台选型（无硬件时）

在购买硬件之前，强烈建议先在仿真中验证：

| 仿真器 | 搭配硬件型号 | 适用场景 |
|-------|------------|---------|
| MuJoCo | G1/H1 官方 MJCF | 算法开发、学术 RL |
| Isaac Lab | G1/H1/H1_2 资产 | 大规模并行训练 |
| Genesis | 新兴，G1 支持中 | 快速原型 |
| Drake | 定制 URDF | 轨迹优化 / MPC 研究 |

---

## 决策树

```
目标是 locomotion（行走/跑步/地形）？
├── 是
│   ├── 需要操作能力 → G1
│   └── 只需移动 → H1（力矩更大）
└── 否（主要做操作 / loco-manip）
    ├── 预算 < 150K CNY → G1
    ├── 预算 OK，需工业可靠性 → Fourier GR1
    └── 仿真先行 → IsaacLab G1 模型
```

---

## 参考来源

- Unitree G1 产品页与 SDK 文档
- Fourier GR1 技术白皮书
- Rudin et al., *Learning to Walk in Minutes* (2022) — 大规模仿真训练基准

---

## 关联页面

- [Locomotion](../tasks/locomotion.md) — 平台选择直接影响 locomotion 任务的难度和方案
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 操作任务需要关节数量和灵巧手支持
- [Sim2Real](../concepts/sim2real.md) — 硬件特性影响 sim2real 策略（SEA vs 刚性关节）
- [Isaac Gym / Isaac Lab](../entities/isaac-gym-isaac-lab.md) — 大规模并行训练平台，支持多种机器人型号
- [legged_gym](../entities/legged-gym.md) — legged_gym 有主流平台的训练模板
