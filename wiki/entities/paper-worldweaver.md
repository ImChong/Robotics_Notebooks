---
type: entity
tags:
  - paper
  - world-models
  - video-generation
  - streaming
  - multi-agent
  - diffusion
  - world-state-registers
  - ucla
  - adobe
status: complete
updated: 2026-07-27
arxiv: "2607.21594"
related:
  - ../overview/world-model-physics-fidelity-outputs.md
  - ./paper-masked-visual-actions.md
  - ./paper-rynnworld-4d-rgb-depth-flow.md
  - ./paper-irasim.md
  - ./paper-vjepa2.md
  - ./paper-dwm-separating-world-effects.md
  - ../methods/generative-world-models.md
  - ../concepts/video-as-simulation.md
sources:
  - ../../sources/papers/worldweaver_arxiv_2607_21594.md
  - ../../sources/repos/worldweaver.md
  - ../../sources/sites/worldweaver-vail-ucla.md
  - ../../sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md
summary: "WorldWeaver / W²（arXiv:2607.21594，UCLA×Adobe）：流式多智能体自回归扩散 + World State Registers；MoT 分路；agent/BEV/text 监督；Minecraft world score 81.0→105.1；代码 checkpoints coming soon（宣称将开源）。"
---

# WorldWeaver（流式多智能体世界状态寄存器 · arXiv:2607.21594）

**WorldWeaver**（\(\mathbf{W}^{\mathbf{2}}\)，*Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers*，[arXiv:2607.21594](https://arxiv.org/abs/2607.21594)，Sicheng Mo* / Yuheng Li* 等 · **加州大学洛杉矶分校（UCLA）** / **奥多比研究院（Adobe Research）**；[项目页](https://vail-ucla.github.io/worldweaver/)，镜像 [vail.cs.ucla.edu/worldweaver](https://vail.cs.ucla.edu/worldweaver/)，[GitHub](https://github.com/VAIL-UCLA/WorldWeaver)）针对 **多智能体 / 多视角** 流式视频世界模型：不只生成各自观测，还维护跨智能体共享、可逐步更新的 **World State Registers（WSR）**，让镜头外演化与跨视角逻辑一致进入显式状态通路。

## 一句话定义

**在流式自回归视频扩散中加入持久可更新的世界状态寄存器，并用 MoT 与多监督信号把共享场景状态从局部帧历史中解放出来。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| W² / WorldWeaver | WorldWeaver | 本文方法名 |
| WSR | World State Registers | 跨智能体持久、逐步提交的寄存器 token |
| MoT | Mixture-of-Transformers | 状态 token 与帧 token 分权重、联合注意力 |
| BEV | Bird’s-Eye View | 俯视布局监督，锚定全局几何 |
| Self-Forcing | — | 用自身生成帧/寄存器滚动，缩小训推差距 |
| DMD | Distribution Matching Distillation 类目标 | Stage3 对齐双向 teacher 分布 |
| KV cache | Key-Value cache | 流式局部窗条件 |
| FID / VLM | Fréchet Inception Distance / Vision-Language Model | 视觉质量与逻辑打分轴 |

## 为什么重要

- **持续状态输出族：** 策展上对应 [物理保真输出轴](../overview/world-model-physics-fidelity-outputs.md) 的 **「持续状态」**——共享场景 / 智能体 / 镜头外信息用寄存器跨片段读写，而不是每步只从近期像素重推世界。
- **多智能体一致性瓶颈：** 单智能体局部 KV 窗口在双玩家 Minecraft 等设定下，难保证双方观测兼容同一 3D 世界；WSR 把「共享状态」提升为一等公民。
- **监督可检查：** agent 状态、BEV、scene text 让寄存器内容可探针，而不是纯隐式记忆黑箱。
- **架构启示：** 状态更新与像素生成抢同一套 dense 权重时会冲突；**MoT 分路** 是可迁移的工程教训。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 加州大学洛杉矶分校（UCLA）、奥多比研究院（Adobe Research） |
| **设定** | 多智能体共享世界；同步预测各视角未来视频 |
| **状态机制** | World State Registers（持久 + 动态更新） |
| **骨干** | 流式自回归 latent 扩散 + **MoT** |
| **监督** | agent status、BEV（DINOv2）、scene text |
| **训练课程** | 双向 teacher → 因果+寄存器 → Self-Forcing |
| **开源** | **宣称将开源 / 占位仓**（截至 2026-07-27） |

## 流程总览

```mermaid
flowchart LR
  subgraph obs [多智能体观测]
    X1[Agent 1 帧窗]
    X2[Agent 2 帧窗]
    A[联合动作]
  end
  subgraph wsr [World State Registers]
    Rprev[r_{i-1}]
    G[更新 Gθ]
    Ri[提交 r_i]
  end
  subgraph gen [流式生成]
    DEN[因果扩散去噪下一 chunk]
    OUT[各视角新帧]
  end
  X1 --> G
  X2 --> G
  A --> G
  Rprev --> G --> Ri
  Ri --> DEN
  X1 --> DEN
  X2 --> DEN
  DEN --> OUT
  OUT --> G
```

## 核心原理

### 问题：只条件历史帧不够

Chunk 自回归扩散每步都要把世界信息从像素里重新「读」出来；局部窗偏好近期可见内容，**镜头外变化与跨智能体约束**容易丢。WorldWeaver 改为显式维护 \(\mathbf{r}_i\)：

\[
\mathbf{r}_i=G_\theta(\mathbf{r}_{i-1},\mathbf{x}_{i-W+1},\ldots,\mathbf{x}_i,a_i),\quad
p_\theta(\mathbf{x}_{i+1}\mid \mathbf{x}_{i-W+1:i},a_{i+1},\mathbf{r}_i).
\]

交错序列 \([x_0,r_0,x_1,r_1,\ldots]\) 使「先提交状态、再生成下一帧」在因果 mask 上成立。

### 寄存器接地

| 监督 | 作用 |
|------|------|
| **Agent status** | 位姿 / 速度 / 朝向 → 局部运动一致 |
| **BEV** | 俯视几何 → 跨视角布局；DINOv2 特征余弦损失 |
| **Scene text** | 语义类别与场景描述 → 可读写语义状态 |

辅助头仅训练期使用，**推理不增加开销**。

### MoT + 三阶段课程

1. **Stage 1：** 双向注意力多玩家 teacher，学同步场景结构。  
2. **Stage 2：** 因果学生 + WSR + \(\mathcal{L}_{\mathrm{flow}}+\lambda\mathcal{L}_{\mathrm{reg}}\)。  
3. **Stage 3：** Self-Forcing，对自生成帧与寄存器滚动，暴露状态漂移。

MoT 让寄存器与帧走不同权重分支，减轻「画清楚 vs 记清楚」的梯度冲突。

## 工程实践

| 项 | 实践要点 |
|----|----------|
| **开源状态** | **宣称将开源**（截至 **2026-07-27**）：[VAIL-UCLA/WorldWeaver](https://github.com/VAIL-UCLA/WorldWeaver) README 写 **Code and checkpoints are coming soon**；项目页有方法与表，**无可运行入口** |
| **项目页** | <https://vail-ucla.github.io/worldweaver/>；机构镜像 <https://vail.cs.ucla.edu/worldweaver/> |
| **复现** | 当前只能读论文 / 项目页；勿假设 clone 可训 |
| **选型** | 需要 **多智能体持续状态** 时跟进本页；单臂操作像素沙盒见 [IRASim](./paper-irasim.md) / [MVA](./paper-masked-visual-actions.md) |

## 源码运行时序图

**不适用**（截至 **2026-07-27**）：官方仓为占位 README，**无可辨识的训练 / 推理脚本或发布权重**；待代码与 checkpoints 实际发布后再补 sequenceDiagram，并同步 [`sources/repos/worldweaver.md`](../../sources/repos/worldweaver.md)。

## 实验与评测

项目页 / 论文以双智能体 Minecraft 视频为主；**world score** 综合视觉与逻辑（越高越好）：

| Variant | World Score ↑（摘录） |
|---------|----------------------|
| Baseline | **81.0** |
| Registers only | **93.8** |
| + Agent stats | 88.1 |
| + BEV | 102.4 |
| + Scene text | 103.2 |
| **+ All（完整 W²）** | **105.1** |

要点：即使无显式目标，寄存器槽位已抬升一致性；BEV 常是最强单信号；全监督增益集中在 grounding / building / consistency 等 **状态敏感** 类别，而非只刷外观 FID。

## 结论

**WorldWeaver 把「共享世界状态」从帧历史里拆出来，用可监督寄存器支撑流式多智能体一致生成；开源尚未落地，先吃方法与指标。**

1. **WSR 是主接口** — 持久跨智能体、逐步提交，条件下一 chunk。
2. **监督决定状态语义** — agent / BEV / text 互补；BEV 强在全局几何。
3. **MoT 缓解目标冲突** — 状态通路与像素通路分权重。
4. **Self-Forcing 必要** — 长时寄存器漂移与帧漂移一起暴露。
5. **工程现状** — **coming soon**；跟踪 GitHub，勿写进可复现清单。
6. **物理保真读法** — 属持续状态族：逻辑一致↑仍须问动作/动力学敏感性是否进入寄存器。

## 局限与风险

- **未开源可运行实现：** 无法验证训练稳定性与超参。
- **域偏 Minecraft 交互：** 迁移到真实机器人多机位仍待证。
- **寄存器容量 / 遗忘：** \(K\) token 瓶颈与长期物体状态更新仍是开放问题。
- **与操作 CEM 规划叙事弱连接：** 主贡献在生成一致性，不直接给 latent 规划 API。

## 与其他工作对比

| 对比轴 | WorldWeaver | [IRASim](./paper-irasim.md) | [V-JEPA 2](./paper-vjepa2.md) | [DWM Separating](./paper-dwm-separating-world-effects.md) |
|--------|-------------|-----------------------------|-------------------------------|----------------------------------------------------------|
| **状态形式** | **显式寄存器 token** | 隐式在视频 latent | 学习视频表征 | 转移分解到 world/action 头 |
| **智能体** | **多智能体同步** | 单臂操作 | 单臂规划 | 单智能体控制基准 |
| **开源** | 占位 | 已开源 | 已开源 | 未开源 |
| **主指标** | World score / 一致性 | 视频质量 + Push-T | SSv2/EK100 + Franka | CEM +13.1 pp |

## 关联页面

- [世界模型物理保真：输出阅读轴](../overview/world-model-physics-fidelity-outputs.md) — **持续状态** 代表
- [IRASim](./paper-irasim.md) / [Masked Visual Actions](./paper-masked-visual-actions.md) — 像素生成对照
- [RynnWorld-4D](./paper-rynnworld-4d-rgb-depth-flow.md) — 几何信号对照
- [V-JEPA 2](./paper-vjepa2.md) — latent 表征对照
- [DWM（Separating World Effects）](./paper-dwm-separating-world-effects.md) — 转移分解对照
- [Generative World Models](../methods/generative-world-models.md)
- [Video-as-Simulation](../concepts/video-as-simulation.md)

## 参考来源

- [WorldWeaver 论文归档（arXiv:2607.21594）](../../sources/papers/worldweaver_arxiv_2607_21594.md)
- [VAIL-UCLA/WorldWeaver 占位仓索引](../../sources/repos/worldweaver.md)
- [WorldWeaver 项目页归档](../../sources/sites/worldweaver-vail-ucla.md)
- [具身智能研究室：世界模型物理保真（微信）](../../sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)

## 推荐继续阅读

- [arXiv:2607.21594](https://arxiv.org/abs/2607.21594)
- [项目页](https://vail-ucla.github.io/worldweaver/)
- [机构镜像](https://vail.cs.ucla.edu/worldweaver/)
- [GitHub — VAIL-UCLA/WorldWeaver](https://github.com/VAIL-UCLA/WorldWeaver)（跟踪代码发布）
