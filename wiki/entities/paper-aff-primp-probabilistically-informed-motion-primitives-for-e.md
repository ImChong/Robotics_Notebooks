---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-11-26
project_id: aff-primp-probabilistically-informed-motion-primitives-for-e
doi: "10.1109/TRO.2024.3390052"
venue: "IEEE T-RO 2024"
summary: "PRIMP（IEEE T-RO 2024）：在 SE(3)/PCG(3) 李群上基于相邻步相对位姿协方差构建无参数轨迹概率密度，支持单演示适应新中间位姿、视角等变、跨机器人工作空间密度融合，并结合 Workspace-STOMP 与物理仿真想象完成真机工具操作（<50 ms 泛化，提速 >5×）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_primp-probabilistically-informed-motion-primitives-for-e.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# PRIMP

**PRIMP: Probabilistically-Informed Motion Primitives for Efficient Affordance Learning from Demonstration**（Sipu Ruan, Weixiao Liu, Xiaoli Wang, Xin Meng, Gregory S. Chirikjian，新加坡国立大学 NUS / 约翰霍普金斯大学，IEEE T-RO 2024 / arXiv:2305.15761v1）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)。本文把李群概率轨迹基元与物理仿真可供性「想象」及引导式运动规划结合，实现少/单演示跨物体、跨机器人与避障操作。

## 一句话定义

一种在机器人 6D 工作空间李群 $SE(3)$（及位姿变化群 $PCG(3)$）上对**相邻时间步相对位姿**建模马尔可夫高斯联合分布的示教学习（LfD）方法，无需基函数或核函数调参即可毫秒级（<50 ms）泛化至新中间位姿、新视角与新机器人，并通过 **Workspace-STOMP** 引导避障规划与仿真可供性想象完成真机工具操作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| PRIMP | Probabilistically-Informed Motion Primitives | 本文提出的基于李群相对位姿分布的轨迹基元方法 |
| GORA | Globally-Optimal Reparameterization Algorithm | 基于变分法的全局最优 $SE(3)$ 示教时间对齐算法 |
| STOMP | Stochastic Trajectory Optimization for Motion Planning | 通过随机 rollout 优化的无梯度运动规划器（本文扩展为 Workspace-STOMP） |
| LfD | Learning from Demonstration | 从人类拖动/遥操作示教中学习技能基元的方法 |

## 为什么重要

- 传统运动基元（如 DMP、ProMP、Orientation-KMP）大多在欧氏/关节空间对**绝对状态**建模，或依赖大量基函数/核参数调优，导致：（1）外推至示教分布外的新起点/途经位姿时轨迹形态易失真；（2）绑定特定机械臂关节空间而难以跨构型迁移；（3）遇到新障碍物时缺乏分布方差引导的避障弹性。
- PRIMP 基于李群相对位姿建模，无需调参且仅靠**单条示教**即可保持轨迹几何形状外推，配合「物理仿真想象（倾倒与舀取可供性关键位姿生成）+ Workspace-STOMP」打通了「可供性感知 $\to$ 动作基元条件生成 $\to$ 无碰撞真机执行」的闭环（详见 [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md) Action 分支）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 作者与机构 | Sipu Ruan, Weixiao Liu, Xiaoli Wang, Xin Meng, Gregory S. Chirikjian（NUS / Johns Hopkins Univ.） |
| 出处 | IEEE Transactions on Robotics (T-RO), 2024 / arXiv:2305.15761v1 |
| DOI / 原文 | `10.1109/TRO.2024.3390052` · <https://doi.org/10.1109/TRO.2024.3390052> / <https://arxiv.org/abs/2305.15761> |
| 方法名 / 分组 | PRIMP · Action（Motion primitives · Explicit affordance · Hierarchical affordance-to-action） |
| 实验平台与框架 | Franka Emika Panda（拖动示教与 5 类真实工具任务）+ Kinova Gen3（跨构型迁移验证）；MoveIt! 插件 + Gazebo 物理仿真想象 |

## 核心机制（归纳）

1. **GORA 全局最优时间重参数化**：对不同速度的人类 $SE(3)$ 示教轨迹，不依赖启发式挑选基准曲线的 DTW，而是通过欧拉–拉格朗日变分法最小化体速度加权总变分 $\int_0^1 \|g^{-1}\partial g/\partial \tau\|_W^2 \dot{\tau}^2 dt$，由累积积分求逆 $\tau^*(t)=F^{-1}(t)$ 解析获得全局最优时间对齐序列。
2. **李群相对位姿联合概率编码（$SE(3)$ 与 $PCG(3)$）**：假设第 $i$ 步偏差仅与相邻步相关（马尔可夫链），从示教相对位姿 $\Delta_{i,i+1}^{(k)} = (g_i^{(k)})^{-1} g_{i+1}^{(k)}$ 估计均值 $\mu_i$ 与局部协方差 $\Sigma_{i,i+1} \in \mathbb{R}^{6\times 6}$；利用伴随算子 $\text{Ad}(\mu_i^{-1}\mu_{i+1})$ 导出整条轨迹李代数偏差 $x_{1,\dots,n}\sim\mathcal{N}(0,\Sigma'_{1,\dots,n})$ 的**稀疏块三对角精度矩阵** $\Sigma'^{-1}_{1,\dots,n}\in\mathbb{R}^{6n\times 6n}$。同步支持完全解耦平移与旋转的 Pose Change Group $PCG(3)=SO(3)\times\mathbb{R}^3$，全程**零超参数**。
3. **三种解析自适应机制**：
   - **带不确定性的途经位姿条件化**：通过李代数 Kalman 滤波式闭式更新，在给定新起点/终点/途经位姿 $g_i^*$ 及协方差 $\Sigma_i^*$ 后解析求得后验分布（即便只有 **1 条示教**配合手工初始协方差也能泛化）。
   - **观察坐标系变换等变性**：当参考系经 $h\in SE(3)$ 变换时，协方差通过 $\text{Ad}^{-1}(h)\Sigma_{i,i+1}\text{Ad}^{-T}(h)$ 严格等变。
   - **跨机器人工作空间密度（Workspace Density）融合**：通过 $SE(3)$ 群卷积 $\rho_{wd}=(\rho_1*\dots*\rho_m)$ 近似目标机器人的末端可达概率密度，再按高斯融合公式将参考轨迹拉向新机器人（如 Franka Panda $\to$ Kinova Gen3）的高可操作区域。
4. **Workspace-STOMP 避障规划与机器人想象**：在 MoveIt! 中以 PRIMP 均值轨迹初始化 STOMP，对关节空间随机 rollout $q$ 做前向运动学得到末端轨迹，计算其到 $m_r$ 条 PRIMP 采样轨迹的平均 $SE(3)$ 旋转+平移距离作为引导代价——在协方差小（任务关键段）紧贴示教形状，在协方差大处灵活绕开未知障碍物。结合 RGB-D 扫描重建与 Gazebo 粒子倾倒/舀取仿真（**机器人想象**）自动生成新物体的功能关键位姿，在真实 Franka Panda 上完成 5 类日常任务。

### 方法流程图

```mermaid
flowchart LR
  Demo["人类拖动示教 SE(3) 轨迹"] --> GORA["GORA 变分法全局时间对齐"]
  GORA --> LieDist["SE(3) / PCG(3) 相对位姿协方差与稀疏精度矩阵"]
  Scan["RGB-D 场景重建 -> Gazebo 物理想象 / ArUco 关键位姿"] --> Adapt["解析自适应: 途经位姿条件化 + 视角等变 + 机器人工作空间密度融合"]
  LieDist --> Adapt
  Adapt -->|"参考轨迹分布样本 + 均值初值"| WSTOMP["Workspace-STOMP 避障轨迹优化 (MoveIt!)"]
  WSTOMP -->|"可行则执行 / 不可行则重采样候选关键位姿"| Robot["Franka Panda 真机执行 (5 类工具操作)"]
```

## 评测与指标

- **编码与泛化计算效率（Abstract & Sec. VII-A）**：PRIMP 编码与条件泛化平均耗时 **< 50 ms**（毫秒级），比对比方法（ProMP、Orientation-KMP）**快 5 倍以上（> 5×）**；同时生成轨迹到示教集与目标新途经位姿的距离比对比方法**接近 2 倍以上（> 2×）**。
- **LASA 手写体基准（26 个单峰类别，$\alpha=0.05$ 临界差异图，Fig. 7）**：在 $D_{\text{via}}$（新途经位姿适应距离）平均排名上，**PRIMP-PCG 位居第 1（1.4831）**、**PRIMP-SE 位居第 2（1.5723）**，显著优于 `KMP-10`（3.2008）、`ProMP`（3.8869）、`KMP-0.1`（5.0408）和 `KMP-0.01`（5.8162）。
- **仿真与真机日用任务 LfD 基准（Fig. 8 & Fig. 10）**：在 MoveIt 字母写画（N/U/S）及 5 类真实拖动示教任务（倒颗粒、搬运、舀颗粒、拉抽屉、开铰链门；每任务 5–10 条示教、测试 50 组新目标/途经位姿）上，$D_{\text{demo}}$ 与 $D_{\text{via}}$ 全面优于 ProMP 与不同核宽度的 Orientation-KMP（`PRIMP-PCG` 因平移与旋转解耦误差不互串，略优于 `PRIMP-SE`）。
- **引导运动规划（Sparse / Cluttered / Narrow 场景，Fig. 14–15 & Table II）**：在空旷/稀疏场景中 `Workspace-STOMP` 达到 **100%** 成功率（Task 1 Sparse 4 个子场景全 100%，而仅用均值轨迹引导的 `Cartesian-STOMP` 在部分场景跌至 40%–85%），规划耗时与无引导 `STOMP` 相当，同时显著缩小了与参考动作分布的偏差。
- **Franka Emika Panda 真机系统验证（Sec. VI）**：使用与示教时**不同形状/材质的工具与容器**并叠加新障碍物，成功完成全部 5 类物理任务（Task 1 倾倒、Task 2 运送、Task 3 舀取、Task 4 开抽屉、Task 5 开门）。

## 与其他工作对比

- **相比 DMP / ProMP / Orientation-KMP**：无需设定基函数数量或核函数带宽超参数；因采用**相邻步相对位姿协方差**而非绝对位姿协方差，当起点或途经位姿大幅外推至示教包络之外时仍能保持整体几何形状不变（Fig. 11），且天然具备视角等变性与跨臂迁移能力。
- **相比 Cartesian-STOMP**：Cartesian-STOMP 仅以单条确定性均值轨迹为参考代价，缺乏局部不确定度信息；Workspace-STOMP 利用 PRIMP 输出的多条分布采样自动在非关键段放宽偏离以避障，在狭窄障碍场景下兼顾成功率与动作保真度。
- 更多同类基于基元或模仿学习的动作策略可参见 [imitation-learning.md](../methods/imitation-learning.md) 与 [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)。

## 常见误区或局限

1. **可供性「想象」仍为手工预编程仿真原语**：Gazebo 中的倾倒与舀取想象逻辑（如勺子先侧转 $90^\circ$ 再上翻计数粒子）是预先编写的物理试探脚本，尚未直接融合人类演示来学习新任务类型的功能位姿。
2. **原始均值轨迹不保证平滑，依赖 STOMP 后处理**：PRIMP 输出的均值/采样路径处于运动学工作空间，有时平滑度不足且不考虑关节限位/障碍碰撞，必须经由 Workspace-STOMP 优化后才能下发真机。
3. **对示教异常值敏感且未建模动力学与力信息**：框架未内置示教离群轨迹自动过滤机制；状态仅含 6D 位姿（未纳入速度、加速度与接触力/力矩），且后验推导假设局部偏差较小以线性化李代数对数映射。

## 结论

**PRIMP 把李群相对位姿概率建模、工作空间密度卷积与引导式采样规划打通为一套「无需调参、毫秒级外推、可跨臂复用」的示教–规划框架，为可供性引导工具操作提供了紧凑可靠的底层运动基元。**

1. **用「相对位姿协方差」替代「绝对位姿协方差」是外推不失真的根本**：起点大幅平移或途经点落在示教包络外时，内部局部几何连接仍保持不变，并支持单条示教冷启动。
2. **$PCG(3)$ 与 $SE(3)$ 的工程选型清晰**：纯机械臂末端轨迹规划优先用 $PCG(3)$（分离旋转与平移减少耦合误差，LASA 排第 1）；若系统受非完整/动力学耦合约束（如车辆、四旋翼），则选 $SE(3)$。
3. **分布式引导规划优于单均值轨迹引导**：利用 PRIMP 采样分布构建 Workspace-STOMP 代价，使规划器能在大方差自由段主动绕障、在小方差关键段严守工具操作姿态。
4. **注意上下游边界**：功能位姿发现目前仍靠 ArUco 标记或预编程 Gazebo 想象脚本，且缺少力/速度建模，接触密集或强受限任务仍需外加视觉泛化或柔顺/力控层。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[imitation-learning.md](../methods/imitation-learning.md)

## 参考来源

- [`sources/papers/affordance_awesome_primp-probabilistically-informed-motion-primitives-for-e.md`](../../sources/papers/affordance_awesome_primp-probabilistically-informed-motion-primitives-for-e.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://doi.org/10.1109/TRO.2024.3390052>

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://doi.org/10.1109/TRO.2024.3390052)
- [arXiv 预印本 (arXiv:2305.15761)](https://arxiv.org/abs/2305.15761)
