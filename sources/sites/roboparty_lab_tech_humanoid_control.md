# TeCH：基于对比表征学习的时间距离建模，用于人形机器人全身控制

> 来源归档（ingest · RoboParty Lab 成果页）

- **URL：** <https://lab.roboparty.com/outputs/tech>
- **标题：** TeCH：基于对比表征学习的时间距离建模，用于人形机器人全身控制
- **机构：** 机器人派对（RoboParty）· RoboParty Lab
- **类型：** paper / lab output
- **入库日期：** 2026-07-14
- **一句话说明：** RoboParty Lab 提出的无监督人形全身控制方法：在 TLDR 对比时间距离表征框架上，用隐空间距离变化构造密集奖励，实现零样本运动跟踪与目标到达；仿真与真机均优于或对标 BFM-Zero / SONIC，且训练成本降近两个数量级。

## 核心摘录（策展，非全文）

### 问题与定位

- 人形高自由度使可扩展运动学习复杂；PPO + 手工奖励（如 SONIC）需 **128 GPU × 3 天** 级资源，且状态覆盖与泛化受限。
- 无监督表征学习两条主线：**Forward-Backward（FB）**（BFM-Zero）与 **对比时间距离（TLDR）**。
- FB 依赖 **linear MDP** 假设，在富接触场景表征不稳定、易退化；TLDR 放弃值分解，直接学习时间可达性。

### 方法要点（TeCH）

1. **时间距离表征：** 对转移 $(s_t, s_{t+1})$ 用 roll 构造伪目标 $g_t$，编码器 $\phi_\psi$ 映射到隐空间；对比损失 $L_{TE}$ 分离时间相距较远状态、平滑相邻步。
2. **密集进度奖励：** $r_{tldr}(s_t, s_{t+1}, z) = \|z - \bar{z}(s_t)\|_2 - \|z - \bar{z}(s_{t+1})\|_2$，驱动策略朝隐目标靠近。
3. **预训练闭环：** 在线交互 → replay → 多目标优化（判别器 → 编码器 → critic → actor）；隐变量 $z$ 混合 **随机 / 在线可达 / 专家轨迹** 三种来源；按时间可达性动态调整目标采样。
4. **真机迁移：** 非对称训练、域随机化、辅助稳定性目标、风格判别器（遵循 BFM-Zero 18 实践）。
5. **零样本推理：** 目标到达 $z_g = \phi(s_g)$；轨迹跟踪 $z_t = \sum_{t'=t}^{t+H} w_{t'} \phi(s_{t'})$。

### 实验设置

- **仿真：** IsaacLab · Unitree G1（29 DoF）；200 Hz 仿真 / 50 Hz 控制。
- **数据：** LAFAN1（~40 段，训练）；100-Style（3–4k 段，测试）；均重定向到 G1。
- **指标：** 关节 MAE $E_{mae}$；EMD 作动作分布匹配消融。

### 主要结果（页面数字）

| 对比 | 结论 |
|------|------|
| **跟踪 vs SONIC** | TeCH **0.1318±0.0329**（LAFAN1）/ **0.1474±0.0405**（100-Style）；SONIC 全轨迹 **0.2916** / **0.1522**；SONIC (TER.) **0.1081** / **0.1355** |
| **跟踪 vs BFM-Zero** | TeCH 略优，尤其 **全局根部旋转漂移**（快速 360° 旋转） |
| **目标到达** | 与 BFM-Zero 精度相当，TeCH **更高效到达** |
| **训练效率** | 单卡 GPU + 数小时 LAFAN1 vs SONIC 128 GPU + 大规模数据 |
| **真机** | 高动态跟踪、目标到达、外力扰动后自主站起（SONIC 难恢复） |
| **随机隐采样** | 无条件 rollout 产生连贯类人运动 |
| **隐空间插值** | 线性插值对应平滑物理一致运动 |

### 消融（默认超参）

- 隐维度 **D=256**（D≥256 后 EMD 饱和；TeCH 各 D 下 EMD 低于 BFM-Zero）
- 编码器学习率 **lr=8×10⁻⁷**
- 在线隐目标更新 **update-z-every=10**

### 工程落点

- 集成于 [Roboparty/UFO](https://github.com/Roboparty/UFO) 无监督 RL 框架（MJLab backend）。
- 页面提供抗扰动与运动跟踪实验视频入口。

## 对 wiki 的映射

- 新建实体页：[`wiki/entities/paper-tech-humanoid-control.md`](../../wiki/entities/paper-tech-humanoid-control.md)
- 交叉更新：[`wiki/entities/roboparty-ufo.md`](../../wiki/entities/roboparty-ufo.md)、[`wiki/overview/roboparty-lab-party-os-technology-map.md`](../../wiki/overview/roboparty-lab-party-os-technology-map.md)
- 谱系对照：[`wiki/entities/paper-bfm-zero.md`](../../wiki/entities/paper-bfm-zero.md)、[`wiki/methods/sonic-motion-tracking.md`](../../wiki/methods/sonic-motion-tracking.md)、[`wiki/concepts/behavior-foundation-model.md`](../../wiki/concepts/behavior-foundation-model.md)

## 参考来源（原始）

- 成果页：<https://lab.roboparty.com/outputs/tech>
- TLDR 原论文：<https://arxiv.org/abs/2407.08464>
- BFM-Zero：<https://arxiv.org/abs/2511.04131>
- SONIC：<https://arxiv.org/abs/2511.07820>
