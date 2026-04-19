---
type: query
tags: [debugging, sim2real, locomotion, rl, deployment, real-robot]
status: complete
summary: RL 策略仿真中表现良好但真机失效时的系统排查手册，覆盖训练问题、部署问题、硬件问题三条排查路径。
sources:
  - ../../sources/papers/sim2real.md
related:
  - sim2real-deployment-checklist.md
  - ../concepts/sim2real.md
  - ../tasks/locomotion.md
  - ../concepts/domain-randomization.md
  - ../concepts/privileged-training.md
---

# RL 策略真机调试 Playbook

> **Query 产物**：本页由以下问题触发：「RL 策略在仿真中好但真机差，如何系统排查？」
> 综合来源：[Sim2Real](../concepts/sim2real.md)、[Domain Randomization](../concepts/domain-randomization.md)、[Sim2Real 部署清单](sim2real-deployment-checklist.md)

## TL;DR 排查决策树

```
策略在真机上失效
│
├─ 是否立刻摔倒 / 完全不工作？
│   ├─ 是 → 先检查【部署问题】（obs 顺序、action 映射、归一化）
│   └─ 否 → 继续向下
│
├─ 行为有一定逻辑但不稳定 / 漂移 / 抖动？
│   ├─ 抖动明显 → 检查控制频率、PD 增益、延迟
│   ├─ 漂移/偏转 → 检查 IMU 标定、质量不对称
│   └─ 性能随时间退化 → 检查热保护、电池电压
│
└─ 行为整体可以但 sim 指标无法复现？
    └─ 进入【训练问题】排查（DR 范围、观测噪声）
```

---

## 症状分类树

### 类别 A：训练问题（策略本身没学好）

| 症状 | 根因假设 | 验证方法 |
|------|---------|---------|
| 真机步速上限远低于仿真 | 力矩限制在仿真中未正确施加 | 对比仿真和真机 torque log |
| 速度指令响应迟钝 | 观测归一化 scale 过大 | 打印原始策略输入，对比仿真 |
| 复杂地形仿真好、真机差 | Domain Randomization 覆盖范围不足 | 对照真机测量值是否超 DR 边界 |
| 策略对扰动极度敏感 | 未加观测噪声随机化 | 仿真中加入等量噪声后测试 |
| 姿态恢复能力差 | 训练初始化太窄，未覆盖意外姿态 | 检查 reset 分布，扩大 initial pose DR |

**排查步骤：**
1. 在仿真中注入真机实测的 IMU 噪声量级，观察策略是否退化
2. 逐项缩小 DR 范围并重测，定位哪个随机化参数影响最大
3. 检查策略训练时的 episode return 分布——是否存在高方差（说明 DR 过宽）

---

### 类别 B：部署问题（策略对但接口错）

| 症状 | 根因假设 | 验证方法 |
|------|---------|---------|
| 上机瞬间倒地 | obs/action 维度顺序错位 | 打印第一帧 obs 并与仿真对照 |
| 关节运动方向反 | action 符号约定不一致 | 单关节手动发送正/负指令验证 |
| 原地不动或微颤 | 控制频率与策略假设不符 | 打印实际执行频率，核查 dt |
| 步态极小或极大 | velocity command 归一化系数错 | 对比仿真中 cmd scale 与真机接口 |
| 关节达到硬件限位 | 仿真关节范围 ≠ 真机 URDF 设置 | 对比 URDF joint limits 与真机手册 |

**排查步骤：**
1. **逐字段核对 obs 向量**：打印仿真和真机的第一帧 obs，逐 index 比对
2. **静止测试**：机器人支撑架上，送零速度指令，观察各关节是否保持静态平衡姿态
3. **单轴验证**：固定其他关节，逐一测试每个关节的方向/幅度映射
4. **推理延迟测试**：测量从 obs 读取到 action 发送的实际延迟，应 < 5ms

---

### 类别 C：硬件问题（传感器/执行器不符合仿真假设）

| 症状 | 根因假设 | 验证方法 |
|------|---------|---------|
| 步态向一侧持续偏移 | IMU 安装偏置或轴向定义错误 | 静止水平放置，读取 IMU roll/pitch |
| 关节力矩饱和、发热 | 真机摩擦远大于仿真设定 | 测量关节空载摩擦力矩 |
| 高频抖振 | 编码器分辨率低导致 obs 量化噪声 | 降低策略输出频率或加低通滤波 |
| 电池电量低时策略失效 | 电压跌落导致力矩响应变慢 | 充满电后重测 |
| 关节到位精度差 | PD 增益 Kp/Kd 与仿真不对应 | 对比 PD 增益，做阶跃响应测试 |

**排查步骤：**
1. 静止放置，记录 30 秒 IMU 数据，检查 drift 和 bias
2. 做单关节 step response：送固定目标角度，记录实际轨迹，与仿真对比
3. 检查真机 URDF 质量/惯量参数是否经过实测校准（尤其大修后）

---

## 常用诊断命令 / 工具

```bash
# 打印第一帧 obs（Isaac Lab / legged_gym 风格）
python play.py --checkpoint <path> --log_obs --num_steps 1

# 测量推理延迟（以 PyTorch 为例）
import time, torch
t0 = time.perf_counter()
action = policy(obs)
print(f"inference: {(time.perf_counter()-t0)*1000:.2f} ms")

# 检查关节顺序是否与 URDF 一致
python -c "import yourrobot; print(yourrobot.joint_names)"

# 记录真机 obs 并保存（用于与仿真对比）
ros2 topic echo /robot/obs --no-arr > obs_log.txt
```

**推荐工具链：**
- **rerun.io**：实时可视化关节角、力矩、IMU，与仿真曲线叠加对比
- **ROS2 bag**：完整记录一个 episode 的所有话题，离线回放分析
- **Matplotlib / Pandas**：绘制 obs/action 时序曲线，对比仿真 rollout

---

## 快速排查优先级

1. 先验证【部署问题】——obs/action 接口是最常见的翻车点，修复成本最低
2. 再验证【硬件问题】——一次静态测试可以排除大多数传感器异常
3. 最后才怀疑【训练问题】——重训练成本高，先确认接口无误再做判断

---

## 参考来源

- [Sim2Real 源文档](../../sources/papers/sim2real.md)
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021)
- Margolis et al., *Rapid Locomotion via Reinforcement Learning* (2022)
- [Sim2Real 部署清单](sim2real-deployment-checklist.md)

## 关联页面

- [Sim2Real 部署清单](sim2real-deployment-checklist.md) — 与本页互补，按阶段列出检查项
- [Sim2Real](../concepts/sim2real.md) — sim2real gap 的来源与缓解方法
- [Locomotion](../tasks/locomotion.md) — 足式任务的评价指标与挑战
- [Domain Randomization](../concepts/domain-randomization.md) — DR 参数设计指导
- [Privileged Training](../concepts/privileged-training.md) — Teacher-Student 框架对 sim2real 的帮助
