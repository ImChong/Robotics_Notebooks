# FastGrasp: Learning-based Whole-body Control method for Fast Dexterous Grasping with Mobile Manipulators（arXiv:2604.12879）

> 来源归档（ingest）

- **标题：** FastGrasp: Learning-based Whole-body Control method for Fast Dexterous Grasping with Mobile Manipulators
- **类型：** paper / mobile manipulation / dexterous grasping / whole-body control / tactile feedback / reinforcement learning
- **arXiv abs：** <https://arxiv.org/abs/2604.12879>
- **arXiv HTML：** <https://arxiv.org/html/2604.12879v1>
- **PDF：** <https://arxiv.org/pdf/2604.12879>
- **项目页：** <https://taoheng-star.github.io/fastgrasp-page/>
- **作者：** Heng Tao*、Yiming Zhong*、Zemin Yang*、Yuexin Ma†（* equal；† corresponding）
- **机构：** 上海科技大学（ShanghaiTech University）
- **硬件：** Agilex Bunker Mini 移动底盘 + Dobot CR5 六轴臂 + LeapHand 16-DoF 灵巧手；RealSense D435i 腕部 RGB-D；每指/掌心 **9 路** 薄膜压力传感 → **二值触觉**；Jetson AGX Orin 机载推理
- **仿真：** NVIDIA Isaac Sim；60 Hz 物理、**15 Hz** 策略；48 并行环境；PPO
- **入库日期：** 2026-07-12
- **一句话说明：** 两阶段学习框架——**CVAE + PointNet** 从物体点云生成多样抓取候选，经 **GWC/GDC 包络度** 选最优引导；**PPO 全身 RL** 同步控制底盘/臂/手，并以 **二值触觉** 在冲击接触中实时收紧抓取；仿真 **50.09%** 平均成功率（全点云），真机高速 **32%** / 降速 **34.62%**。

## 摘要级要点

- **痛点：** 静态灵巧操作受固定基座限制；移动操作多限于平行夹爪；触觉增强系统算力高或响应慢，难支撑 **高速动态抓取** 下的冲击稳定、全身实时协调与跨物体泛化。
- **两阶段框架：** (1) **抓取引导生成**——条件 VAE 从点云采样 **150** 个候选，经前向可达锥 + 碰撞过滤 + **GWC/GDC** 质量排序选最优；(2) **带触觉的全身策略学习**——观测含本体、底盘速度、腕摄物体位姿、**二值触觉** 与引导姿态，PPO 输出底盘速度 + 臂/手关节目标。
- **训练数据：** 抓取生成器在 **478,200** 条合成抓取（4,782 物体 × 100）上训练；策略在 **418** 类物体（来自 RealDex / DexGrasp Anything 系）上训练，分 easy/hard 评测。
- **仿真主结果（Table I，全点云）：** 本文 **50.09%** S.R. vs 反应式移动操作 [3] **8.31%**、单阶段 **3.03%**、力方向两阶段 **41.99%**；**部分点云** 下本文 **38.51%**，基线近乎归零。
- **真机（Table III）：** 完整配置（DR + LPF + TA）在高速 **32%**、半速 **34.62%**；缺 DR/LPF **0%**；缺触觉适应 **26.87% / 29.62%**。
- **Sim2Real：** 腕摄 RGB-D 点云对齐、**低通滤波**（α=0.3）平滑指令、**域随机化**、**触觉适应**（据接触反馈收紧关节）。

## 核心摘录（面向 wiki 编译）

### 1) 系统与平台（§III）

| 子系统 | 规格 |
|--------|------|
| 移动底盘 | 前进最大 **1.3 m/s**，偏航 **1.0 rad/s**，0.7 s 静止加速至满速 |
| 机械臂 | 5 可控关节（第 4 关节水平锁定）+ 位置控制，最大 **100°/s** |
| 灵巧手 | LeapHand **16-DoF**，最大 **90°/s** |
| 感知 | 腕部 D435i 点云；策略 **15 Hz** 与仿真一致 |
| 触觉 | 薄膜压力 → **二值接触**（掌 + 各指段），利于实时推理与 sim2real |

### 2) 抓取引导：CVAE + GWC/GDC（§IV-B–C）

- **CVAE：** PointNet 编码点云特征 $\mathbf{F}$，解码器从 $\mathcal{N}(0,I)$ 采样 $\mathbf{z}$ 生成抓取 $\mathbf{g}$（位置、旋转、16 维手关节）。
- **前向过滤：** 相对机器人–物体法向定义 **前向抓取锥**，剔除需绕物体 **>90°** 的后向抓取。
- **GWC / GDC：** 在拇指–各指宽度轴与掌法向深度轴上，度量指尖区间与物体投影区间的 **覆盖率**；综合 $\text{Quality}=(\sum_k \text{GWC}_k\cdot\text{GDC}_k)\cdot\text{GDC}_{thumb}$，无需显式表面法向，适合噪声/部分点云。

### 3) 全身 PPO 策略（§IV-D）

- **观测：** 臂/手关节、底盘速度、腕摄物体位置与掌距、二值触觉 $\mathbf{c}_t$、引导抓取 $\mathbf{g}$。
- **动作：** 底盘 $(v_f, v_y)$ + 臂 5-DoF 关节位置 + 手 16-DoF 关节位置。
- **奖励层次：** 臂展半径、底盘速度/朝向、预抓取对齐、手型开合、快速接近、稳定持握时长、**触觉接触计数**、动作平滑惩罚等十项；训练曲线分 **locomotion → preparation → execution** 三阶段收敛。
- **优化：** PPO + GAE。

### 4) 实验与消融（§V）

| 设置 | 要点 |
|------|------|
| 仿真 unseen | 全点云 easy/hard **59.50% / 34.42%**；部分点云 **47.17% / 24.09%** |
| 奖励消融 | 去触觉观测 **−15.65%**；去触觉奖励 **−11.56%** |
| 引导消融 | 随机引导 **14.45%**；仅 GDC **34.57%**；仅 GWC **18.90%** |
| 真机 16 物体 | Simple 10 + Complex 6；每物体 **10** 次连续试验 |

### 5) 与近邻工作对照

| 工作 | 关系 |
|------|------|
| [UniDexGrasp](https://arxiv.org/abs/2303.04163) | 多样抓取提案 + 目标条件策略；本文扩展到 **移动全身 + 高速** |
| Catch it! [32] | 移动双手 **空中接物**；平行夹爪，缺多样物体精细抓取 |
| Burgess-Limerick [3] | 反应式移动抓取；物体简化为单点，无点云/灵巧手 |
| DexGrasp Anything [33] | 同实验室物理感知灵巧抓取；**静态** 设定 |
| DexH2R [26] | 同作者组动态灵巧交接 benchmark |

## 对 Wiki 的映射

- [FastGrasp（移动灵巧快速抓取）](../../wiki/entities/paper-fastgrasp-mobile-dexterous-grasping.md) — 论文实体归纳页
- [Manipulation](../../wiki/tasks/manipulation.md) — 灵巧抓取与触觉闭环
- [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 移动底盘 + 臂 + 手的全身协调
- [抓取专题](../../wiki/overview/topic-grasp.md) — 高速动态抓取锚点
- [Tactile Sensing](../../wiki/concepts/tactile-sensing.md) — 二值压力触觉与冲击稳定

## BibTeX（arXiv 页）

```bibtex
@article{tao2026fastgrasp,
  title={FastGrasp: Learning-based Whole-body Control method for Fast Dexterous Grasping with Mobile Manipulators},
  author={Tao, Heng and Zhong, Yiming and Yang, Zemin and Ma, Yuexin},
  journal={arXiv preprint arXiv:2604.12879},
  year={2026}
}
```

## 当前提炼状态

- [x] 摘要与 §III–V 机制对齐
- [x] wiki 页面映射确认
- [x] 项目页入口索引
