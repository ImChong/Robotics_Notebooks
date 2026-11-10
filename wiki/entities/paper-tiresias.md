---
type: entity
tags:
  - paper
  - world-model
  - language-conditioned
  - benchmark
  - planning
  - diffusion
status: complete
updated: 2026-11-10
project_id: tiresias-lcwm
arxiv: "2611.10392"
code: https://github.com/UWMILab/Tiresias
related:
  - ../concepts/world-action-models.md
  - ../concepts/functional-taxonomy-world-models.md
  - ../concepts/video-as-simulation.md
  - ../methods/diffusion-policy.md
  - ../methods/model-predictive-control.md
  - ../concepts/diffusion-transformer.md
  - ../entities/awesome-world-models.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/tiresias_arxiv_2611_10392.md
original:
  - ../../sources/papers/tiresias_arxiv_2611_10392.md
summary: "Tiresias（arXiv:2611.10392）：提出语言条件世界建模任务 LCWM 与跨四域基准 LCWM-Bench，并用扩散策略出 128 条候选、轨迹排序器留 8 条、视频世界模型想象后由视觉排序器融合选 1 条；全部为离线评测，未做真机。"
---

# Tiresias / LCWM-Bench（arXiv:2611.10392）

**Tiresias**（*Tiresias: Towards Language-Conditioned World Models*，[arXiv:2611.10392](https://arxiv.org/abs/2611.10392)）来自华盛顿大学、新加坡国立大学、Microsoft Research 与卡内基梅隆大学（论文首页）。它同时给出一个任务（LCWM）、一个基准（LCWM-Bench）和一个基线智能体（Tiresias）。

## 一句话定义

**从一帧第一视角图像和一句语言指令出发，先让语言条件扩散策略采样大量候选动作序列，用不看图像的轨迹排序器筛出少数几条，再让视频世界模型只对这几条想象后果，由视觉排序器与轨迹分数融合选出最终计划（§4）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LCWM | Language-Conditioned World Modeling | 本文提出的任务：初始观测 + 指令 → 满足指令的动作序列（§3.1） |
| LCWM-Bench | — | 配套基准：15 个数据集、4 个域、三种指令风格（§3） |
| LDP | Language-conditioned Diffusion Policy | 仅以指令为条件的 DiT 扩散策略，出候选轨迹（§4.1） |
| Tr-Ranker | Trajectory Ranker | 只看指令与动作序列，预测 SR / nDTW / ATE / RPE 并合成分数（§4.3） |
| V-Ranker | Visual Ranker | 对世界模型想象出的视频潜变量打分（§4.3） |
| WM | World Model | 以 Wan2.2 为骨干的视频扩散世界模型（§4.2） |
| SR | Success Rate | 预测终点与参考终点距离在域内平均步长内即成功；机器人另要求夹爪终态一致（附录 C.1） |
| nDTW | normalized Dynamic Time Warping | 轨迹整体对齐程度 |
| ATE / RPE | Absolute Trajectory Error / Relative Pose Error | 全局漂移 / 局部精度误差 |
| MPC | Model Predictive Control | 本文中 Tiresias (MPC) 指用采样式 MPC 取代 LDP 出候选的变体（§5.1） |

## 为什么重要

- 导航、机器人操作、3D 游戏、驾驶各自有不同的动作接口、数据集与指标，语言条件世界模型很难横向比较；LCWM-Bench 用统一的动作参数化与指标把它们放到同一协议下（§1、§3）。
- 给出一个可复现的「先筛后想象」范式：想象（世界模型 rollout）是贵的，所以只对筛后的少数候选做（§4.4、§5.3 b3）。
- 本页价值主要在评测设置与推理预算的数字，而不是真机能力：论文自己承认没有真机实验（附录 D）。

## 核心原理

### 流程总览

```mermaid
flowchart LR
  A[初始观测 + 语言指令] --> B[LDP 采样 N=128 条候选轨迹]
  B --> C[Tr-Ranker 打分 只看语言与动作]
  C --> D[保留 top-k=8 条]
  D --> E[世界模型对每条想象未来视频]
  E --> F[V-Ranker 给视觉分]
  C --> G[分数组内归一化后加权融合 2比1]
  F --> G
  G --> H[选最高分的 1 条作为最终计划]
```

### 各模块（作者自述）

- **LDP（§4.1、B.2）**：DiT 扩散策略，条件只有 umT5 指令嵌入（每个 block 内交叉注意力）和扩散步，视觉输入被刻意排除，使不同噪声种子给出多样的、符合指令的候选；从零训练。
- **世界模型（§4.2）**：Wan2.2 初始化的多模态扩散 Transformer；候选轨迹经轻量 MLP 直接条件化去噪器，初始帧的 DINOv3 特征与 umT5 指令嵌入、每域系统提示通过交叉注意力拼接；初始帧 VAE 潜变量也作为锚点；训练目标为 flow matching（式 3）。
- **Tr-Ranker / V-Ranker（§4.3、B.4–B.5）**：Tr-Ranker 用交叉注意力 + 时间 Transformer，四个头预测 SR（交叉熵）与 nDTW / ATE / RPE（Huber）；V-Ranker 对想象视频潜变量取时间均值、标准差与帧间差等描述子，加指令、场景和域嵌入，四个头按 1:0.5:0.5:0.25 合成 S_v，且不直接读取动作序列。
- **选择（式 4）**：argmax λ_tr·S_tr + λ_v·S_v，两分数在 shortlist 内归一化，默认 λ_tr:λ_v = 2:1；四个模块独立训练，推理时不联合微调（§4.4、B.6）。

### 基准构建（§3）

- 15 个数据集：导航 4 个（GoStanford、RECON、HuRoN、TartanDrive）、机器人 3 个（含 BridgeData V2、RT-1、LIBERO）、3D 游戏 2 个（含 Minecraft）、驾驶 4 个（含 nuScenes、Waymo、DrivingDojo）；混合真实与仿真来源（§3.2）。
- 观测统一为 256×256；位移按各来源平均步长归一；过滤后退与过短片段（§3.2）。
- 142,987 条 episode、约 4.4M 帧、428,961 条评测样本；机器人占 40%、导航 29%、游戏 21%、驾驶其余（§3.3）。
- 划分：train / val seen / val unseen / test；val unseen 含完全不在训练中的来源（如导航的 TartanDrive）（§3.2）。

## 实验结果（论文原数）

| 项 | 数值 | 出处 |
|----|------|------|
| 总分（SR 与 nDTW 宏平均） val seen / val unseen | Tiresias 63 / 51；Tiresias (MPC) 59 / 46 | 表 1 |
| 最强对照 | val seen：LCVN-WM 54；val unseen：LCVN-Uni 42 | 表 1 |
| 无语言的 NWM vs 加语言的 NWM (lang) | seen 45 → 48；unseen 33 → 35 | 表 1 |
| 想象质量 val seen（Tiresias vs LCVN-WM） | FVD@32 274.0 vs 321.6；SSIM@10 0.546 vs 0.289 | 表 2 |
| LDP 条件 | 仅语言：不筛选 SR 0.13、全流程 0.60；仅观测：0.06 / 0.41；语言 + 观测：0.11 / 0.56 | 表 4 |
| 8 条随机候选的选择器 | 随机 0.13；Tr-Ranker 0.30；V-Ranker 0.28；融合 0.39；Oracle-of-8 0.52 | 表 5 |
| 128 条池里留 k 条的保留率 | k=1 / 4 / 8 / 16：0.68 / 0.80 / 0.85 / 0.86 | 表 6 |
| 换提案方法后 SR（不筛选 → Tr-Ranker → Tr + V） | MPC：0.11 → 0.49 → 0.56；NoMaD (lang)：0.32 → 0.42 → 0.47；LDP：0.13 → 0.53 → 0.60 | 表 7 |
| 延迟（val seen） | 128→8→1：SR 0.60，36 s；128→2→1：SR 0.55，9 s；NoMaD (lang)：0.32，0.7 s；LCVN-Uni：0.47，85 s | 表 A5 |
| 指令风格 | Landmark 最好，Concise 略优于 Intricate | 图 5 |

## 工程实践

- 默认配置：N=128、k=8、λ_tr:λ_v=2:1；世界模型在 LCWM-Bench 上训练 10 个 epoch，8×A100 80GB；DINOv3 ViT-L/16 distilled（300M）、umT5、VAE 冻结（§5.1、C.2）。
- 预算分配读法：把便宜的筛选（不渲染）放前面，把贵的想象放后面。按表 A4 / 表 6，128→1 的 SR 为 0.530，128→8 为 0.602，N 增到 256 收益很小；表 6 中 k 从 8 增到 16 只涨 0.01。预算该花在哪一级，可参考这些曲线（来源 §5.3 b2–b4、表 A4）。
- 仓库 README（2026-11-10 核对）列出 `ldp/train.py`、`lcwm/main.py`、`rankers/tr_ranker/train.py`、`rankers/v_ranker/train.py` 四个训练入口，需自行申请 DINOv3 权重；是否可复现数字：**未核实**。

## 局限

**论文自述**

- 没有在实体机器人上部署 Tiresias，真机效果待验证（附录 D）。
- 所有方法在 val unseen 上总分更低，对未见来源的泛化仍难（§5.2）。
- 残余错误：后段轨迹漂移（指令未限定运动幅度时），以及想象未来看似合理但偏离真实环境、超出初始观测可推断的部分（§5.3 c1）。
- 最终选择与 Oracle-of-N 之间仍有明显差距，且 Oracle-of-N 随池增大继续上升而 SR 已饱和（§5.3 b2、表 5）。

**本页读者判断（推测或无出处处标注）**

- SR 的定义是预测轨迹的积分终点与参考轨迹终点的距离在平均步长内（附录 C.1），并非在环境里执行后检查任务是否完成；机器人域额外比对夹爪终态。因此「成功」是对参考轨迹的贴合程度，不等于闭环任务成功。
- 基准里「机器人」只是 3 个数据集（含真机 Bridge / RT-1 与仿真 LIBERO）的离线评测；机器人 Tiresias 的 val unseen SR 为 0.39，驾驶为 0.40（表 1），距可用仍远。
- 36 s 的单次规划延迟（表 A5，论文未写硬件之外的细节）说明它更像离线规划或评测工具，不是实时控制器（推测）。
- 与基线的差距部分来自世界模型骨干更大（Wan2.2 视频模型对比 Diamond、NWM 等）；论文未做同规模对照（推测）。
- 表 1 / 表 2 只汇报 val seen 与 val unseen；基准含 test 划分，但未见 test 结果，README 数据目录也没列 test。
- 本页未复现任何数字，也未运行代码。

## 与强化学习的关系

- 论文没有用强化学习；没有奖励、没有策略梯度，LDP 是对专家轨迹的监督去噪训练（§4.1），排序器是监督回归 / 分类（B.4）。
- 对 RL 的相关性是旁证：世界模型 + 候选采样 + 打分选择，与基于模型的规划（采样式 MPC，论文自己也把它做成 Tiresias (MPC) 基线）同属一类思路；见 [模型预测控制](../methods/model-predictive-control.md)。

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [世界模型功能分类](../concepts/functional-taxonomy-world-models.md)
- [视频即仿真](../concepts/video-as-simulation.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [Diffusion Transformer](../concepts/diffusion-transformer.md)
- [模型预测控制](../methods/model-predictive-control.md)
- [Awesome World Models](../entities/awesome-world-models.md)
- [VLA 方法页](../methods/vla.md)

## 参考来源

- [sources/papers/tiresias_arxiv_2611_10392.md](../../sources/papers/tiresias_arxiv_2611_10392.md)
- [arXiv:2611.10392](https://arxiv.org/abs/2611.10392)
- 代码：<https://github.com/UWMILab/Tiresias>

## 继续阅读

- [arXiv PDF（v1）](https://arxiv.org/pdf/2611.10392v1)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [World Action Models](../concepts/world-action-models.md)
