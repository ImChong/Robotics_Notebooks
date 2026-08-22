# LooperMuscle（arXiv:2608.00820）

> 来源归档（ingest）

- **标题：** LooperMuscle: Fast and Stable Learning of Humanoid Whole-Body Tracking via Structured Mixture-of-Experts
- **类型：** paper / humanoid / whole-body-tracking / reinforcement-learning / mixture-of-experts / FastSAC
- **arXiv abs：** <https://arxiv.org/abs/2608.00820>
- **PDF：** <https://arxiv.org/pdf/2608.00820>
- **HTML：** <https://arxiv.org/html/2608.00820>
- **项目页：** <https://loopermuscle.github.io/>
- **机构：** 深镜智能（DeepMirror Inc.，广州）；香港科技大学（HKUST）；穆罕默德·本·扎耶德人工智能大学（MBZUAI）；通讯 Xingxing Zuo（`xingxing.zuo@mbzuai.ac.ae`）
- **作者：** Boyi Liu、Qijin Li、Tianqi Yu、Qinrui Yan、Xingxing Zuo
- **发表 / 上传：** 2026-08-01（arXiv v1）
- **硬件：** Unitree G1（29 DoF）；真机 50 Hz ONNX 经 Holosoma 栈
- **训练栈：** 论文主表 MJLab（4096 envs，50 Hz 控制）；真机策略在 Holosoma 可部署 154-D 观测接口上 **从头重训**
- **入库日期：** 2026-08-22
- **一句话说明：** 在 FastSAC 墙钟加速脉络上，用 **语义分组 MoE actor + 专家感知分布式 critic + 配额路由 replay/延迟课程** 闭环，把 29-DoF 全身跟踪从 FastSAC-MLP 的 15 min 低质量区拉到约 **45 min 接近 PPO 6 h 质量**（40 条 LAFAN1 上 body err ↓34%）。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2608.00820](https://arxiv.org/abs/2608.00820) | 论文与附录 |
| 项目页 | [loopermuscle.github.io](https://loopermuscle.github.io/) | 演示与 Code 按钮 |
| 代码 | [LooperMuscle/Code](https://github.com/LooperMuscle/Code) | Holosoma 训练/推理/重定向 |
| 基线算法 | [FastSAC 15-min](https://arxiv.org/abs/2512.01996) | off-policy 墙钟加速前驱 |
| 质量上界 | PPO | 论文 6 h 参考 |
| 对照 MoE | [GMT](https://arxiv.org/abs/2506.14770)、[KungfuBot2](https://arxiv.org/abs/2509.16638) | PPO+MoE 并发工作（Table IV） |
| 仿真器 | [MJLab](https://arxiv.org/abs/2601.22074) | 论文定量基准 |
| 真机栈 | [Holosoma](https://github.com/amazon-far/holosoma) | 可部署观测接口与运行时 |
| 数据 | LAFAN1（40 seq） | Walk/Run/Jump/Dance/Fight/Fall&GetUp |

## 开源状态（步骤 2.5，截至 2026-08-22）

- **部分开源：** 项目页链到 [LooperMuscle/Code](https://github.com/LooperMuscle/Code)（Apache-2.0）。README 写明提供 **部署栈**（Holosoma 推理、MuJoCo 评测、G1 真机 WBT）、**运动重定向** 与 **Holosoma 训练框架**（`train_agent.py`，支持 PPO/FastSAC）。
- **边界：** 论文 Table I–III 的 **MJLab 特权观测基准**（含仿真器 ground-truth anchor）与 Holosoma 可部署接口 **不同**；真机策略需在 Holosoma 154-D 接口上重训，**不直接迁移 MJLab checkpoint**。
- **待核实：** 论文声称「released codebase」含 MJLab 超参；仓内主训练入口为 Holosoma/MJWarp/IsaacGym，MJLab 实验脚本见 `demo_scripts/` 与 `paper_assets/experiments/`（以仓内 README 为准）。

## 摘要级要点

- **问题：** FastSAC 类 off-policy 方法把人形训练压到 ~15 min，但在 **全身跟踪（WBT）** 上跟踪质量明显弱于 PPO（~6 h），形成持续的速度–性能鸿沟。
- **根因（作者）：** 单体策略难兼顾上下身异质动力学；标量 critic 难以细粒度 credit assignment；均匀 replay 让强专家主导梯度、弱专家停滞——三者相互强化。
- **方法：** LooperMuscle = **语义分组 MoE actor**（$K{=}4$，$G{=}2$ 上下身门控 + KL 负载均衡）+ **专家感知分布式 critic**（每专家 DVF 头，门控加权聚合）+ **配额路由 replay + 延迟解锁课程**（按专家贡献路由、渐进释放难样本）。
- **闭环：** Actor → 贡献向量 → 路由 → Critic → 梯度反哺专家特化。
- **主结果（40 LAFAN1，MJLab 特权接口）：** LooperMuscle body err **0.101 m** vs FastSAC-MLP **0.153 m**（↓34%），norm. reward **0.723** vs **0.648**；墙钟 **~45 min** vs PPO **~360 min**（norm. reward 1.0）。
- **消融：** 去 MoE actor +51.5% body err；去 quota replay +25.7%；去 deferred scheduling +17.8%；去 expert-aware critic +11.9%。
- **真机：** G1 上跟踪 KungfuBot2 动作库格斗序列；Holosoma 可部署接口重训后验证 sim-to-real。

## 核心摘录（面向 wiki 编译）

### 1) MoE Actor（§III-B）

\[
a_{t,j}=\sum_{k=1}^{K}w_{k,g(j)}(\mathbf{s}_{t})\;c_{k}\;S_{k,j}\;\mu_{k,j}(\mathbf{s}_{t})
\]

- 组级门控 $w_{k,g}$（温度 $\tau_g$ 可分组）；$S_{k,j}$ 按关节名义范围初始化；$c_k$ 输出对齐。
- 负载均衡：$\mathcal{L}_{\text{lb}}=\lambda_{\text{lb}}\alpha_{\text{lb}}(t)\,\text{KL}(\bar{\mathbf{w}}\|\mathbf{u})$。

### 2) Expert-Aware Distributional Critic（§III-C）

- 每 critic、每专家 categorical DVF（C51 原子）；用 actor 门控 $\tilde{w}_k$ 聚合，结构与 actor 同构。

### 3) Quota-Routed Replay（§III-D）

- 转移附带贡献向量 $\mathbf{e}_t$、body tracking ratio $b_t$；按主导专家 $k^*$ 路由；$N_k=\lfloor q_k N_{\text{exp}}\rfloor$ 保证配额。
- 延迟桶：holdout $h(\rho)$ 与 unlock $u(\rho)$ 随训练进度释放难样本。

### 4) 主表（Table I，40 seq）

| Method | Body Err. (m) | Joint Err. (rad) | Norm. Reward | Time |
|--------|---------------|------------------|--------------|------|
| PPO | 0.082 ± 0.031 | 0.243 ± 0.089 | 1.000 | ~360 min |
| LooperMuscle | 0.101 ± 0.038 | 0.285 ± 0.102 | 0.723 | ~45 min |
| FastSAC-MLP | 0.153 ± 0.057 | 0.401 ± 0.134 | 0.648 | ~15 min |

### 5) 实验设定（§IV-A）

- MJLab，G1 29-DoF，4096 envs，50 Hz 控制，单卡 RTX 4090D。
- 40 LAFAN1：Walk 12 / Run 6 / Jump 3 / Dance 8 / Fight 5 / Fall&GetUp 6。
- **完整性披露：** 主表用 MJLab 特权 anchor 观测；真机用 Holosoma 154-D 可部署接口重训。

## 对 wiki 的映射

- 沉淀实体页：[LooperMuscle（论文实体）](../../wiki/entities/paper-loopermuscle.md)
- 交叉补强：[FastSAC 15-min](../../wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md)、[FlashSAC](../../wiki/methods/flashsac.md)、[GMT](../../wiki/entities/paper-gmt.md)、[Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)、[人形策略网络架构](../../wiki/concepts/humanoid-policy-network-architecture.md)、[人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[PPO vs SAC](../../wiki/comparisons/ppo-vs-sac.md)

## 当前提炼状态

- [x] arXiv 摘要 / 方法 / Table I–III / 实验设定摘录
- [x] 项目页与 GitHub 开源核查（步骤 2.5）：**部分开源**（Holosoma 栈 + 训练/部署；MJLab 基准与特权接口需区分）
- [x] wiki 实体页与交叉链接
