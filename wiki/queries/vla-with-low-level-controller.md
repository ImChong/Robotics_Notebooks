---
type: query
tags: [vla, wbc, mpc, impedance-control, architecture, fusion, latency, manipulation, humanoid]
status: complete
summary: "VLA 如何与低级关节控制器（MPC/WBC）融合：梳理3种主流架构（VLA+PD、VLA+阻抗、VLA+WBC异步）、输出格式选择、action chunking 缓冲策略与实践挑战。"
sources:
  - ../../sources/papers/rl_foundation_models.md
related:
  - ../methods/vla.md
  - ../concepts/whole-body-control.md
  - ../methods/model-predictive-control.md
  - ../queries/vla-deployment-guide.md
---

# Query：VLA 与低级关节控制器（MPC/WBC）融合架构

> **Query 产物**：本页由以下问题触发：「VLA 如何与低级关节控制器（MPC/WBC）融合？有哪些架构？」
> 综合来源：[VLA](../methods/vla.md)、[Whole-Body Control](../concepts/whole-body-control.md)、[MPC](../methods/model-predictive-control.md)、[VLA 真机部署指南](../queries/vla-deployment-guide.md)

## TL;DR 核心结论

| 架构 | VLA 输出 | 低层控制器 | 延迟容忍 | 安全性 | 控制带宽 | 复杂度 |
|------|---------|---------|---------|-------|---------|-------|
| **VLA + PD** | 关节位置目标 | PD 位置控制器 | 低（< 20 ms） | 低 | 高频可行 | 低 |
| **VLA + 阻抗控制** | 末端 EE 位姿 / 刚度目标 | 阻抗/导纳控制 | 中（20~80 ms） | 中 | 中频（50~200 Hz） | 中 |
| **VLA + WBC 异步** | EE 目标姿态 / 任务命令 | WBC + 逆动力学 | 高（50~200 ms） | 高 | WBC 内部高频（1 kHz） | 高 |

**选型建议**：
- 纯操作任务（桌面）：VLA + 阻抗控制，平衡安全与实现复杂度
- 人形全身操作 / 移动操作：VLA + WBC 异步，WBC 处理平衡约束
- 简单快速原型验证：VLA + PD，上手最快

---

## 详细内容

### 1. 为什么 VLA 不能直接驱动执行器

VLA 的核心挑战在于时域错配：

| 维度 | VLA 特征 | 执行器需求 | 矛盾 |
|------|---------|---------|-----|
| 推理频率 | 5~20 Hz（GPU 推理延迟 50ms+） | 关节控制 500~1000 Hz | 差 100 倍 |
| 输出稳定性 | 受网络随机性影响，帧间可能抖动 | 需要平滑连续的力矩命令 | 直接驱动会引起抖振 |
| 安全保证 | 无显式约束，黑盒端到端 | 需要关节限位、力矩饱和、碰撞保护 | VLA 可能输出超限命令 |
| 接触感知 | 依赖视觉，盲区内失效 | 接触瞬间需要力控响应（< 5 ms） | VLA 响应太慢 |

因此，无论选择哪种架构，**VLA 都应该作为中高层策略，而不是 1 kHz 力矩控制器**。

---

### 2. 架构一：VLA + PD 位置控制

#### 2.1 架构图

```
VLA（5~20 Hz）
    ↓ 目标关节角度 q_des（或 delta q_des）
PD 位置控制器（500~1000 Hz）
    τ = Kp × (q_des - q) + Kd × (q̇_des - q̇)
    ↓ 关节力矩
执行器
```

#### 2.2 VLA 输出格式

- 绝对关节角度：$q_{\text{des}} \in \mathbb{R}^n$（需要与仿真训练一致的关节顺序）
- 增量关节角度：$\Delta q \in \mathbb{R}^n$（相对当前状态的偏移，更安全）
- Action chunk：一次输出未来 $K$ 步的关节角序列 $[q_t, q_{t+1}, ..., q_{t+K-1}]$

#### 2.3 优缺点

**优点**：
- 实现最简单，PD 控制器是所有关节驱动的最小配置
- 推理延迟低时（< 20 ms）性能表现良好
- 适合训练数据本身就是关节轨迹的情况（ACT、BC 等）

**缺点**：
- 无接触力控制能力，接触任务容易过力损坏硬件
- VLA 输出抖动会直接传导到关节，无柔顺缓冲
- 不处理平衡约束，人形机器人上使用需额外稳定模块

---

### 3. 架构二：VLA + 阻抗/导纳控制

#### 3.1 架构图

```
VLA（5~20 Hz）
    ↓ 末端目标位姿 x_des（EE pose）或 刚度目标 K_des
阻抗控制器（200~500 Hz）
    F = K × (x_des - x_ee) + D × (ẋ_des - ẋ_ee)
    τ = J^T × F
    ↓ 关节力矩（含接触柔顺性）
执行器
```

#### 3.2 VLA 输出格式

- **EE 绝对位姿**：$x_{\text{des}} = [\text{position} \in \mathbb{R}^3, \text{orientation} \in SO(3)]$
- **EE 增量位姿**：$\Delta x_{\text{des}}$（相对于当前末端的偏移，更适合 delta action 输出）
- **可变刚度目标**：$K_{\text{des}}$（VLA 同时输出目标位姿和期望刚度，实现自适应柔顺性）

#### 3.3 阻抗参数设计建议

| 任务类型 | 刚度 $K$ | 阻尼 $D$ | 说明 |
|---------|---------|---------|-----|
| 自由空间移动 | 高（500~1000 N/m） | 适中 | 快速响应，精确跟踪 |
| 接触建立阶段 | 低（50~200 N/m） | 较高 | 柔顺着陆，避免过力 |
| 装配 / 插孔 | 极低（20~80 N/m） | 高 | 最大柔顺性，容许位置误差 |
| 持续施力阶段 | 中（100~300 N/m） | 适中 | 平衡力控与位置保持 |

#### 3.4 优缺点

**优点**：
- 接触任务安全性高，柔顺性天然吸收 VLA 输出抖动
- EE 位姿是 VLA 最自然的输出格式（人手演示直接对应末端轨迹）
- 对 VLA 推理延迟（50 ms 以内）有一定容忍度

**缺点**：
- 逆运动学（IK）或逆动力学（ID）计算增加系统复杂度
- 刚度参数需要针对任务手动调整
- 仍不处理全身平衡，人形机器人需配合质心控制器

---

### 4. 架构三：VLA + WBC 异步

#### 4.1 架构图

```
VLA（5~20 Hz）
    ↓ 任务命令（EE 目标 / 技能 token / 运动指令）
任务调度层 / Action Buffer（中间层）
    - 对 VLA 输出做插值 / 平滑 / 安全过滤
    ↓ 连续任务空间参考轨迹
WBC + MPC（200~1000 Hz）
    - CLF-CBF-QP 稳定性和安全性保证
    - TSID / HQP 全身力矩分配
    - 平衡 + 操作联合优化
    ↓ 关节力矩
执行器
```

#### 4.2 VLA 输出格式

- **高层技能 token**：VLA 输出离散技能代码（"grasp"/"place"/"push"），WBC 负责执行对应技能原语
- **EE 目标序列**：$[x_{\text{des},0}, x_{\text{des},1}, ..., x_{\text{des},K}]$（action chunk），WBC 做插值执行
- **质心 + EE 联合目标**：同时指定 CoM 高度、身体姿态和末端位姿（人形全身操作场景）

#### 4.3 Action Buffer 缓冲策略

VLA 推理延迟 50~200 ms，WBC 每 1 ms 需要参考命令，必须有缓冲机制：

```python
# 伪代码：Action Chunk Buffer
class ActionBuffer:
    def __init__(self, horizon=16, dt=0.001):
        self.chunk = None        # VLA 输出的 action chunk
        self.ptr   = 0           # 当前执行到 chunk 的哪一步
        self.dt    = dt          # WBC 控制频率步长

    def update(self, new_chunk):  # 被 VLA 异步更新（低频）
        self.chunk = new_chunk
        self.ptr   = 0

    def get_current_ref(self):    # 被 WBC 同步调用（高频）
        if self.chunk is None or self.ptr >= len(self.chunk):
            return self.fallback_ref()   # 超时则返回保持/回退目标
        ref = self.chunk[self.ptr]
        self.ptr += 1
        return ref

    def fallback_ref(self):       # VLA 迟到时的安全降级
        return hold_current_pose()
```

#### 4.4 优缺点

**优点**：
- 最高的安全性：WBC 在底层强制执行关节限位、接触力约束、平衡
- 对 VLA 推理延迟容忍度最高（action chunk 可缓冲 0.5~2 s 的延迟）
- 支持全身协调（人形步行操作、移动抓取等复杂任务）

**缺点**：
- 系统最复杂，需要 WBC 实现、任务空间映射、接触调度等大量工程工作
- WBC 本身需要精确动力学模型，sim2real gap 挑战更大
- VLA 输出格式和 WBC 接口之间需要精心设计任务约定

---

### 5. VLA 输出格式详细对比

| 输出格式 | 典型模型 | 适合低层控制器 | 优势 | 劣势 |
|---------|---------|-------------|------|------|
| 关节角目标 $q_{\text{des}}$ | ACT、BC | PD 控制 | 简单直接，训练数据直接对应 | 无接触柔顺，抖动直接传导 |
| EE 绝对位姿 | RT-2、π₀ | 阻抗控制、IK | 直观，适合演示数据 | 需要 IK，接近奇异点时不稳定 |
| EE 增量位姿 $\Delta x$ | Octo、ACT-delta | 阻抗控制 | 更平滑，隐式安全（小步长） | 累积误差，长时序漂移 |
| 技能 token | 分层 VLA | WBC 技能原语 | 高层抽象，延迟容忍最高 | 技能库设计复杂，覆盖不全 |
| action chunk 序列 | π₀、Diffusion | 任意低层控制器 + buffer | 延迟容忍，动作平滑 | Chunk 边界处可能有跳变 |

---

### 6. 实践挑战与解决方案

#### 6.1 Sim2Real 问题

VLA 通常在仿真或人工演示数据上训练，与 WBC/MPC 联合使用时 sim2real gap 更复杂：

- **VLA 层**：视觉 domain gap（相机参数、光照、背景）
- **WBC 层**：动力学 domain gap（摩擦、刚度、延迟）

解决方向：
- VLA 使用真实演示数据 fine-tune（优先解决视觉 gap）
- WBC 使用 Domain Randomization 覆盖物理 gap（见 [domain-randomization-guide.md](./domain-randomization-guide.md)）

#### 6.2 安全保证

VLA 是黑盒模型，直接驱动执行器存在输出异常的风险：

| 安全机制 | 实现位置 | 说明 |
|---------|---------|-----|
| 关节速度限幅 | 低层控制器 | 过滤 VLA 输出的跳变 |
| 工作空间边界 | 任务调度层 | 拒绝超出工作空间的 EE 目标 |
| 力矩饱和 | WBC / 驱动器 | 硬件保护 |
| CBF 安全过滤 | WBC 层 | 可证明的安全集维持（见 [clf-cbf-in-wbc.md](./clf-cbf-in-wbc.md)） |
| Fallback 策略 | Action Buffer | VLA 失效 / 超时时切换到保守控制 |

#### 6.3 延迟管理

整体系统延迟来源：

```
视觉采集（5~33 ms）
    + 图像预处理（1~5 ms）
    + VLA 推理（30~200 ms）
    + 动作解码（1~2 ms）
    + Action Buffer 延迟（0~chunk_length × dt）
    + WBC 求解（0.5~2 ms）
    + 电机通信（1~5 ms）
─────────────────────────────
总延迟：50~250 ms（取决于 VLA 规模和 GPU 配置）
```

关键措施：GPU 推理异步化（不阻塞控制线程）+ Action Chunk 覆盖推理延迟。

---

## 参考来源

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (2023) — VLA 架构与输出格式
- Black et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control* (2024) — Action chunk + 低层控制器结合
- [sources/papers/rl_foundation_models.md](../../sources/papers/rl_foundation_models.md) — RT-1/RT-2/π₀/Octo 综述
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT, 2023) — Action chunking 缓冲机制

## 关联页面

- [VLA（方法概念）](../methods/vla.md) — VLA 架构总览与核心优缺点
- [Whole-Body Control](../concepts/whole-body-control.md) — WBC 框架、TSID/HQP 实现细节
- [Model Predictive Control](../methods/model-predictive-control.md) — MPC 与 VLA 的组合模式
- [Query：VLA 真机部署指南](../queries/vla-deployment-guide.md) — 部署 checklist 与延迟管理实践
