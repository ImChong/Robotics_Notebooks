# Content Backlog

本文档是 `Robotics_Notebooks` 的内容待办区。

**不是**所有想加的内容都值得现在加。按优先级排队，做一个打一个。

---

## 优先级说明

| 优先级 | 含义 |
|--------|------|
| P0 | 当前主线必须补，补完才能继续往前走 |
| P1 | 对完整性有明显提升，不补也基本能用 |
| P2 | 锦上添花，当前阶段可以暂缓 |

---

## P0 — 主线补全（当前最高优先）

### 补关键实体页

把“概念和方法”接到“工具和生态”。

| 待办 | 位置 | 说明 |
|------|------|------|
| ~~`Isaac Gym / Isaac Lab`~~ ✅ | `wiki/entities/isaac-gym-isaac-lab.md` | 已完成：主流 RL 训练仿真环境，人形/足式必熟 |
| ~~`MuJoCo`~~ ✅ | `wiki/entities/mujoco.md` | 已完成：最常用的机器人仿真器之一 |
| ~~`legged_gym`~~ ✅ | `wiki/entities/legged-gym.md` | 已完成：ETH 开源四足/人形 RL 训练框架 |
| `Unitree` | wiki/entities/ | 国内最主流人形/四足硬件平台之一 |
| `Crocoddyl` | wiki/entities/ | CNRS 的飞踢式人形优化控制框架 |
| ~~`Pinocchio`~~ ✅ | `wiki/entities/pinocchio.md` | 已完成：刚体动力学 C++ 库，TSID/WBC 常用底层 |

### 补缺失概念页

当前主线还有少量缺口：

| 待办 | 位置 | 说明 |
|------|------|------|
| `Floating Base Dynamics` | wiki/concepts/ | 浮动基系统建模，人形控制核心前置 |
| `Contact Dynamics` | wiki/concepts/ | 接触力学基础，locomotion 关键 |
| `Gait Generation` | wiki/tasks/ | 步态生成方法，和 LIP/MPC 强相关 |
| `LQR / iLQR` | wiki/methods/ | 经典控制方法，MPC 的基础 |
| `Capture Point / DCM` | wiki/concepts/ | 比 ZMP 更适合高动态的稳定性指标 |

---

## P1 — 完善性补全

### 扩展任务页

| 待办 | 位置 | 说明 |
|------|------|------|
| `Loco-manipulation` | wiki/tasks/ | 边走边操作，人形机器人实用方向 |
| `Balance Recovery` | wiki/tasks/ | 扰动恢复，跌倒检测与自恢复 |
| `Jump / Hopping` | wiki/tasks/ | 跳跃控制，高动态 locomotion 延伸 |

### 扩展方法页

| 待办 | 位置 | 说明 |
|------|------|------|
| `Policy Optimization Methods (PPO/SAC/TD3)` | wiki/methods/ | 当前主流机器人 RL 算法 |
| `Diffusion Policy` | wiki/methods/ | 机器人操作领域最活跃方向之一 |
| `Model-Based RL` | wiki/methods/ | 补全 RL 方法论中 model-based 这块 |
| `Privileged Training` | wiki/methods/ | sim2real 里常用的训练方式 |

### 扩展比较页

| 待办 | 位置 | 说明 |
|------|------|------|
| `LIP vs Centroidal Dynamics` | wiki/comparisons/ | 建模层次的选型决策 |
| `Direct vs Indirect Control` | wiki/comparisons/ | 更基础的控制方法选型 |

---

## P2 — 长期扩展

以下属于未来扩展方向，当前不急：

- Perception / Vision-based Control
- SLAM / Navigation
- ROS2 / Middleware Integration
- VLA (Vision-Language-Action Models)
- Humanoid Whole-body Manipulation
- Real Hardware Deployment Pipeline
- 感知融合（视觉 + IMU + 力控）

---

## 新增页面的最低质量标准

每个 backlog 条目完成后，必须满足：

1. **一句话定义**在最前面（5 秒内知道这页在讲什么）
2. **为什么重要**说明动机（为什么要学这个）
3. **核心内容**覆盖：定义、公式直觉（如果有）、典型应用、局限
4. **关联页面**：至少链接到 2 个相关 wiki 页
5. **推荐继续阅读**：至少 1 个外部资源
6. **必须被 `index.md` 或对应模块页索引**

不满足以上标准的页面，不允许合入 main。

---

## Backlog 维护规则

- 每次新增内容后，同步更新本文档（删掉已完成的）
- 新想法先放 backlog，不要直接开新文件
- P0 条目完成后才能高优先级处理 P1

---

## 当前 P0 清单

- [x] Isaac Gym / Isaac Lab 页面
- [x] MuJoCo 页面
- [x] legged_gym 页面
- [ ] Unitree 页面
- [ ] Crocoddyl 页面
- [x] Pinocchio 页面
- [ ] Floating Base Dynamics 页面
- [ ] Contact Dynamics 页面
- [ ] Capture Point / DCM 页面
