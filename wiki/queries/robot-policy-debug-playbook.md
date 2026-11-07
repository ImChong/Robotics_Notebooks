---
type: query
tags: [debugging, sim2real, locomotion, rl, deployment, real-robot]
status: complete
summary: RL 策略仿真中表现良好但真机失效时的系统排查手册，覆盖训练问题、部署问题、硬件问题三条排查路径。
original:
  - ../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md
  - ../../sources/papers/rma_reading_arxiv_2107_04034_pt3.md
  - ../../sources/papers/rapid_locomotion_rl_arxiv_2205_02824.md
  - ../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md
  - ../../sources/papers/extreme_parkour_arxiv_2309_14341.md
sources:
  - ../../sources/papers/sim2real.md
  - ../../sources/blogs/wechat_shenlan_rl_sim2real_deployment_2026-09-18.md
related:
  - ./rl-sim2real-deployment-conditions.md
  - ./sim2real-checklist.md
  - ../concepts/sim2real.md
  - ../tasks/locomotion.md
  - ../concepts/domain-randomization.md
  - ../concepts/privileged-training.md
  - ../entities/plotjuggler.md
  - ../entities/rerun-io.md
  - ../entities/tensorboard.md
  - ../entities/weights-and-biases.md
  - ../comparisons/wandb-vs-tensorboard.md
updated: 2026-11-07

---

# RL 策略真机调试 Playbook

> **Query 产物**：本页由以下问题触发：「RL 策略在仿真中好但真机差，如何系统排查？」
> 综合来源：[Sim2Real](../concepts/sim2real.md)、[Domain Randomization](../concepts/domain-randomization.md)、[Sim2Real 工程 Checklist](sim2real-checklist.md#快速部署检查)

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
- **[TensorBoard](../entities/tensorboard.md)**：训练期标量、`Episode_Reward/*` 与 PPO/AMP `Loss/*`；本地 `tensorboard --logdir logs/`（见 [AMP_mjlab](../entities/amp-mjlab.md) 曲线判据）
- **[Weights & Biases](../entities/weights-and-biases.md)**：多机 run 对比、Artifacts 与 rollout 视频；与 TB 可并行（选型见 [W&B vs TensorBoard](../comparisons/wandb-vs-tensorboard.md)）
- **[PlotJuggler](../entities/plotjuggler.md)**：拖拽式多曲线、rosbag2/ROS topic/MCAP 与布局保存；适合快速对齐仿真与真机同名 topic 字段（生产优先 3.17.x；多模态/3D 可评估 PJ4 beta）
- **[rerun](../entities/rerun-io.md)**：实时可视化关节角、力矩、IMU / 图像点云，与仿真曲线叠加对比
- **ROS2 bag**：完整记录一个 episode 的所有话题，离线回放分析
- **Matplotlib / Pandas**：绘制 obs/action 时序曲线，对比仿真 rollout

---

## 原文对照：症状背后的一手证据（2026-11-07 核对）

上文的阈值（如「推理延迟 < 5 ms」「上机立即倒地先查 obs 顺序」）是维护者的工程经验，**一手论文并未给出这些数值**（推测性经验）。下面把能在论文里找到证据的排查点列出，供判断优先级时参考：

| 症状 / 排查点 | 一手证据 | 排查含义 |
|---------------|----------|----------|
| 上机后抖动、肢体剧烈颤动 | Hwangbo 2019：用理想或手调解析执行器训练的策略，真机上「一步都走不了」，肢体剧烈抖动，作者归因于各处延迟没建对 | 先怀疑执行器 / 延迟模型，而不是奖励；考虑执行器网络或系统辨识 |
| 仿真好、真机步速上不去 | Margolis 2022 表 IV：指令 6.0 m/s，仿真 5.46 m/s，真机 3.81 m/s（有在线辨识）/ 2.49 m/s（无）；作者承认仿真评估参数（电机强度、摩擦）可能被高估 | 把「仿真评估配置」当成待校准变量，而不是真值 |
| 复杂地形或载荷下表现差 | RMA：同一策略在训练中未见的沙、泥、床垫、油布滑移区、8 kg 以上负载下，带适应模块的版本成功率显著高于无适应版本；无适应的版本在重载下几乎不动但很少摔 | 区分「完全失效」与「保守不动」，后者更像适应缺失而非接口错误 |
| 延迟、时序抖动 | Extreme Parkour 把深度延迟固定为 0.08 s、本体延迟固定 0.016 s（不足则补等待）以消除抖动；DreamWaQ++ 训练里随机化 0–15 ms 本体延迟，作者称再大会让策略过保守；RMA 的适应模块 10 Hz 与策略 100 Hz 异步 | 测量并**固定**部署侧延迟，使其落在训练分布内 |
| 策略对不准确的速度观测敏感 | Hwangbo：对关节速度加 U(−0.5, 0.5) rad/s 的强噪声使策略对速度测量误差更鲁棒；把速度从观测里去掉则训练完全失败 | 观测噪声应以真机量级为准，不要删观测 |
| 同一策略仿真相近、真机相差大 | Hwangbo：不同激活函数的策略在仿真里表现相近，在真机上可能差别很大；有界激活（tanh）在受扰动时动作更温和 | 仿真分数不能代替真机验收 |
| 低增益 / 高增益与 jitter | Margolis 取 Kp=20、Kd=0.5 的低增益促进平滑且不再调；Hwangbo 的 kd 取到足以抑制真机振荡 | 抖动时先核对 PD 增益与训练一致，其次才调 DR |
| 长期运行退化 | Hwangbo 的策略三个月内多次测试期间机器人增重约 2 kg、换了 3 倍硬度弹簧，策略依然可用（论文未涉及热保护 / 电池，本页相关排查属经验） | 区分「硬件漂移」与「接口错误」 |

使用方式：

1. 症状指向「接口」时优先走【部署问题】清单，这一类的优先级判断（obs 顺序 > 硬件 > 训练）是本页的工程经验，原文无直接对照。
2. 症状指向「动力学」（抖动、速度上不去、地面差）时，优先看执行器 / 延迟 / 在线适应，论文都有明确的反例（理想执行器模型训出的策略走不了；无在线辨识时高速差距显著变大）。
3. 论文提醒的共同点：**别用单一仿真指标判断迁移成功**。

## 快速排查优先级

1. 先验证【部署问题】——obs/action 接口是最常见的翻车点，修复成本最低
2. 再验证【硬件问题】——一次静态测试可以排除大多数传感器异常
3. 最后才怀疑【训练问题】——重训练成本高，先确认接口无误再做判断

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| Kp | Proportional Gain | PD 控制的位置误差增益，影响刚度与响应 |
| Kd | Derivative Gain | PD 控制的速度误差增益，抑制振荡 |
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |
| RMA | Rapid Motor Adaptation | 从历史轨迹隐式估计环境参数的快速运动自适应 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 参考来源

- [Sim2Real 源文档](../../sources/papers/sim2real.md)
- Kumar et al., [*RMA: Rapid Motor Adaptation for Legged Robots*](https://arxiv.org/abs/2107.04034) (2021) — [通读归档](../../sources/papers/rma_reading_arxiv_2107_04034_pt3.md)
- Margolis et al., [*Rapid Locomotion via Reinforcement Learning*](https://arxiv.org/abs/2205.02824) (2022) — [通读归档](../../sources/papers/rapid_locomotion_rl_arxiv_2205_02824.md)
- Hwangbo et al., [*Learning Agile and Dynamic Motor Skills for Legged Robots*](https://arxiv.org/abs/1901.08652) (2019) — [通读归档](../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md)
- Radosavovic et al., [*Real-World Humanoid Locomotion with RL*](https://arxiv.org/abs/2303.03381) (2023) — [通读归档](../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md)
- Cheng et al., [*Extreme Parkour with Legged Robots*](https://arxiv.org/abs/2309.14341) (2023) — [通读归档](../../sources/papers/extreme_parkour_arxiv_2309_14341.md)
- [Sim2Real 工程 Checklist](sim2real-checklist.md#快速部署检查)

## 关联页面

- [Sim2Real 工程 Checklist](sim2real-checklist.md#快速部署检查) — 与本页互补，按阶段列出检查项
- [Sim2Real](../concepts/sim2real.md) — sim2real gap 的来源与缓解方法
- [Locomotion](../tasks/locomotion.md) — 足式任务的评价指标与挑战
- [Domain Randomization](../concepts/domain-randomization.md) — DR 参数设计指导
- [Privileged Training](../concepts/privileged-training.md) — Teacher-Student 框架对 sim2real 的帮助
- [ONNX](../entities/onnx.md) — 策略导出格式与 opset 对齐
- [ONNX Runtime](../entities/onnxruntime.md) — 机载 C++ 推理 Session 与 EP 选型
- [ONNX Runtime vs MNN vs TensorRT](../comparisons/onnxruntime-vs-mnn-vs-tensorrt.md) — runtime 横向对比
