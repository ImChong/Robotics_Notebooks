# ReactiveBFM: Reactive Closed-Loop Motion Planning Towards Universal Humanoid Whole-Body Control（arXiv:2606.30362）

> 来源归档（ingest）

- **标题：** ReactiveBFM: Reactive Closed-Loop Motion Planning Towards Universal Humanoid Whole-Body Control
- **缩写：** **ReactiveBFM**
- **类型：** paper / humanoid whole-body control + closed-loop motion planning
- **arXiv：** <https://arxiv.org/abs/2606.30362>
- **PDF：** <https://arxiv.org/pdf/2606.30362>
- **项目页：** <https://xiao-chen.tech/reactivebfm>
- **发表日期：** 2026-06-29
- **作者：** Xiao Chen, Weishuai Zeng, Xiaojie Niu, Zirui Wang, Jianan Li, Huayi Wang, Furui Xu, Jiahe Chen, Weixiang Zhong, Lihe Ding, Kailin Li, Jiangmiao Pang, Tai Wang, Tianfan Xue, Jingbo Wang
- **机构：** 香港中文大学、上海人工智能实验室
- **入库日期：** 2026-07-01
- **一句话说明：** 把 **BFM 类通用跟踪控制器** 与 **自回归运动扩散规划器** 真正闭合成 **实时闭环规划–控制** 系统：用 **scheduled prefix sampling** 从 imperfect 物理状态学误差恢复，用 **异步重规划 + trajectory chunking 时间集成** 解决 AR 规划与 50 Hz 跟踪的时延失配；Unitree G1 真机实现 **文本条件全身运动、流式指令切换、零样本移动目标到达** 与强扰动恢复。

## 核心论文摘录（MVP）

### 1) 问题：BFM 只能开环执行参考，级联规划器会累积 exposure bias（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2606.30362>
- **核心贡献：** 现有 **Behavior Foundation Models（BFM）** 与 universal tracker（如 SONIC）提供强健低层控制先验，但 **只接受精确轨迹输入**；与 TextOp、DART、Kimodo 等生成式规划器 **开环级联** 时，跟踪偏差使规划器迅速偏离训练分布，**累积 exposure bias** 导致级联失败。论文主张：要实现真正 **reactive whole-body coordination**，规划器必须在 **在线闭环** 中根据本体感知持续重规划。
- **对 wiki 的映射：**
  - [ReactiveBFM 论文实体](../../wiki/entities/paper-reactivebfm.md)
  - [Behavior Foundation Model](../../wiki/concepts/behavior-foundation-model.md)
  - [BFM 论文实体](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)

### 2) Reactive Whole-Body Motion Planner：AR-MDM + prefix curriculum（§3.1）

- **链接：** <https://arxiv.org/html/2606.30362>
- **核心贡献：**
  - **紧凑 36-dim 状态表示**：$x_i=[p_i, q_i, \theta_i]$（根平移 3 + 四元数 4 + G1 关节 29），去掉接触态/全局速度等过参数化，缩小误差空间。
  - **AR Motion Diffusion Model（AR-MDM）**：在滑动窗口上自回归预测 40 帧 motion chunk；条件含 **文本、目标位置、历史本体感知 prefix**。
  - **Scheduled prefix sampling curriculum**：训练初期 teacher forcing，随后线性衰减 ground-truth prefix 概率，转入 **self-rollout**；并对 prefix 注入高斯噪声作域随机化，迫使模型从 **不完美物理状态** 学 error-recovery。
  - **重规划平滑**：一阶/二阶 **temporal consistency loss**；对文本与目标位置做 **condition dropout** 学无条件 motion prior，支持流式指令切换。
  - **数据**：100STYLE、AMASS-HumanML3D、Kungfu 经仿真物理校正；另合成 10k PhysHSI 目标到达轨迹 → **37.14 h** 动力学验证数据。
- **对 wiki 的映射：**
  - [Curriculum Learning](../../wiki/concepts/curriculum-learning.md)
  - [Domain Randomization](../../wiki/concepts/domain-randomization.md)
  - [AMASS](../../wiki/entities/amass.md)

### 3) 闭环系统与真机部署：异步重规划 + trajectory chunking（§3.2）

- **核心贡献：**
  - **事件驱动异步重规划**：控制缓冲未执行帧 $< N_{buf}=10$ 时，非阻塞触发规划线程；规划器 TensorRT **19.3 ms**，控制器 **5.9 ms**；控制器进程 **实时 CPU 调度**，维持 **50 Hz** 不被规划抢占。
  - **Trajectory chunking 时间集成**：融合重叠预测 chunk 的位置/旋转参考，消除异步重规划抖动。
  - **零样本移动目标到达**：训练仅用静态到达数据；推理时每步将机器人根姿态重置为原点、目标变换到 **egocentric 帧**，把动态跟踪分解为静态到达子任务，同时用历史 proprioception 保留动量。真机 **10 次试验 90% 成功率**，连续执行 **>40 s**。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md)（低层跟踪基线对照）

### 4) BFM 跟踪控制器（§3.3）

- **核心贡献：** 采用与 SONIC、早期 BFM 同族的 **预训练 universal tracking controller**；在 **>1.02 亿帧、50 FPS** 重定向数据上 PPO 训练，强全局跟踪 reward + 域随机化，对 OOD 状态偏差具韧性，与闭环规划互补。
- **对 wiki 的映射：**
  - [BFM 论文实体](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md)

### 5) 实验结论与局限（§4–5）

- **Sim-to-sim（躯干/骨盆 100N、0.1s 扰动）：** ReactiveBFM 闭环 **93.1% 成功率**、**2.0% 跌倒率**、$E^r_{MPJPE}$ **34.6 mm**、重规划平滑度 **96.9 mm**；显著优于 TextOp+SONIC（76.4% / 14.7%）及开环级联（最高 Kimodo+SONIC 70.4%）；较开环最佳 baseline **+28.6%** 成功率。
- **消融：** 去掉 self-rollout → 70.5%；去掉 temporal loss → 83.0%；dense representation → 89.1%。
- **真机：** 文本条件多样技能（太极、功夫、蝴蝶踢等）、流式多轮文本切换、暴力扰动恢复（踢、3 kg 球击打、拖拽）；HTC Vive Ultimate Tracker 广播移动目标位姿。
- **局限：** 未显式建模全身人–物交互（loco-manipulation 接触丰富任务）；规划器仅条件于文本、紧凑目标与运动学状态，缺视觉/触觉。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-reactivebfm.md`](../../wiki/entities/paper-reactivebfm.md)
- 关联升级：
  - [BFM 论文实体](../../wiki/entities/paper-behavior-foundation-model-humanoid.md) — 补 ReactiveBFM 作为 BFM **闭环规划上层** 代表
  - [Behavior Foundation Model](../../wiki/concepts/behavior-foundation-model.md) — 层次化控制 / goal-conditioned 线补闭环规划
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md) — Learning-based WBC 段补闭环生成规划
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md)（对照基线）
  - [Curriculum Learning](../../wiki/concepts/curriculum-learning.md)（prefix sampling curriculum）

## 其他公开资料

- **项目页：** <https://xiao-chen.tech/reactivebfm> — 真机演示、移动目标、扰动恢复、流式文本控制
- **Citation：** Chen et al., arXiv:2606.30362, 2026

## 当前提炼状态

- [x] 摘要与核心方法摘录（闭环架构、AR-MDM、prefix curriculum、异步重规划、真机指标）
- [x] wiki 实体页规划
- [ ] 项目页 site 归档（可选，后续）
