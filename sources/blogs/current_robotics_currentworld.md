# CurrentWorld-0: A Cross-Embodiment, Multi-View, Force-Tac World Simulator

> 来源归档（blog / Current Robotics 官方）

- **标题：** CurrentWorld-0: A Cross-Embodiment, Multi-View, Force-Tac World Simulator
- **类型：** blog
- **作者：** Current Robotics Team
- **原始链接：** https://current-robotics.com/blog/currentworld
- **发表日期：** 2026-08（公司首页标注 August 2026 · World Model）
- **入库日期：** 2026-08-17
- **抓取方式：** 官方博客页面直接抓取（WebFetch）；公司首页 <https://current-robotics.com/> 交叉核对产品/论文入口
- **一句话说明：** Current Robotics 把 **Curr-0** 缺的评测—纠正—后训练环做成 **CurrentWorld-0**：跨本体、多视角、力/触觉联合的动作条件 **交互世界模拟器**，策略 / Agent / 人类均可接管；失败状态可保存、回滚、分支，用于规模化评测与 Human-in-the-World-Model 后训练。

## 项目页与开源核查（步骤 2.5）

核查日 **2026-08-17**：

| 入口 | 结果 |
|------|------|
| 博客 <https://current-robotics.com/blog/currentworld> | 正文、演示视频叙事、Citation；**无** GitHub / Hugging Face / 数据集 / 权重链接 |
| 公司首页 <https://current-robotics.com/> | 产品为可穿戴采数外骨骼（Head / Hand / Full-Body）；Research 列出 **dWorldEval**、**Hi-WM**（Apr 2026）；**无** CurrentWorld 代码入口 |
| GitHub / HF 检索 `CurrentWorld-0`、`current-robotics` | **未发现** 官方训练/推理仓或权重 |

**开源结论：确认未开源。** 博客未承诺 “code will be released”；勿按可复现基线引用定量图。姊妹论文 [Hi-WM](https://arxiv.org/abs/2604.21741) 为 Awesome 策展索引级条目，同样不以本博客为代码入口。

公司首页列出的相关论文（非本博客正文）：

- dWorldEval: Scalable Robotic Policy Evaluation via Discrete Diffusion World Model（Li / Zhou / Chen / Xue / Zhu，Apr 2026）
- Hi-WM: Human-in-the-World-Model for Scalable Robot Post-Training（Li / Zhou / Chen 等，Apr 2026，[arXiv:2604.21741](https://arxiv.org/abs/2604.21741)）

## 核心摘录（归纳，非全文）

### 问题重框

- 机器人是 **系统问题**：可扩展的 **数据 + 模型 + 评测**；数据飞轮是种子，但能力必须在环境里被持续验证与改进。
- **真机评测** 受机器人台数、工时、场地多样性约束，难规模化。
- **物理仿真**（如 Isaac Sim）可并行、便宜，但对 deformable / 流体 / 铰接 / 接触丰富操纵 / 材料与长尾日常物理 **难以忠实建模**。
- 目标：保留真机多样性与复杂度，同时获得仿真的速度与并行度 → **生成式交互环境**（策略、AI Agent 或人类操作员均可控）。

定义：世界模型 = **interactive world simulator**，从大规模数据学习如何控制机器人并与世界交互；视觉逼真 **不够**，必须在异构观测与控制接口下预测动作后果。

### 三项定义能力

| 能力 | 含义 |
|------|------|
| **Cross-embodiment** | 不同本体与同一世界交互；捕捉跨本体共享的环境动力学 |
| **Multi-view** | 头戴 / 腕部第一人称 + 第三人称外视；联合预测且跨视角一致 |
| **Force-tactile** | 同一物理动力学在视觉、力、触觉上的联合显现 |

### 跨本体：不统一底层动作空间

- 固定双臂、移动双臂、人形 **不共享** 自然低层动作空间。
- CurrentWorld-0 **不要求** 动作空间对齐；每个本体有 **embodiment-specific action subspace**，模型在多平台数据上 **联合训练**。
- 评测三类本体：
  - **Humanoids**：全身动作；末端含 **BrainCo 灵巧手**、**Wuji 灵巧手**、夹爪
  - **Mobile robots**：臂 + 底盘 + 升降
  - **Dual-arm**：桌面任务
- 目标不是把异构机器人压成单一控制表示，而是让 **一个** 世界模型学习并预测差异极大的交互。

### 多视角一致性

- 策略常同时吃头戴、腕部、第三人称相机。
- 联合建模同步多视角，动作条件预测未来；**不是** 各相机独立生成。
- 物体构型、机器人位姿、接触状态、任务进度须跨视角一致。
- 博客演示叙事：篮子取条纹袜叠沙发、桌面取红杯、反复抓放枕头等。

### 力 / 触觉：像素之外

- RGB 对接触起始、作用力、初期滑移等局部接触状态 **弱可观测或不可观测**。
- 同一交互轨迹上联合预测未来视觉 **以及** 力 / 触觉测量演化。

### 能做什么

1. **世界模型内遥操作 / Human-in-the-World-Model**
   - 策略先在生成环境自主执行；失败或失败倾向态由人类接管给纠正动作。
   - 中间状态可 **保存、回滚、分支** 多条恢复轨迹，无需在真机反复复现失败。
   - 接管轨迹直接用于 **post-training**；博客称各设定成功率上升，难任务增益更大；若干原策略全失败设定在后训练后出现成功 rollout。
   - 评测策略族叙事：**π0、π0.5、DP**（Diffusion Policy）。
2. **与真机一致的策略评测**
   - 博客称世界模型内成功率与真机 **强相关**，并保持相对排名；生成 rollout 复现真机主要失败模式。
   - 把反复版本对比与失败诊断移入世界模型；真机评测集中在最有信息量的策略 / checkpoint / 失败案例。

### 训练中世界建模如何涌现

对比不同 checkpoint 输出（而非只看 loss 曲线）：

1. **动作一致性最先出现** — 跨视角视觉细节仍不稳，但机器人运动方向、粗轨迹、时间进程已一致。
2. **静态世界随后稳定** — 背景、空间布局、应静止物体停止漂移，给运动提供空间参照。
3. **物理交互最后出现** — 接触导致物体如何动、运动如何展开、物体响应是否与动作耦合。

叙事：**先学动作，再稳住场景，最后学二者之间的交互。**

### 与 Curr-0 的闭环

- **Curr-0**：学在物理世界行动的 loco-dexterous 策略。
- **CurrentWorld-0**：策略被执行、评测、纠正、规模化改进的环境。
- 二者互补：**策略学习行动；世界模型让这些行动可规模化评测、纠正与改进。**
- 目标不是取代真机，而是让 **两次真机部署之间** 的迭代大幅可扩展。

## 对 wiki 的映射

- [current-robotics-currentworld](../../wiki/entities/current-robotics-currentworld.md)（系统实体 + Mermaid 闭环图）
- 姊妹：[Curr-0](../../wiki/entities/current-robotics-curr0.md)
- 交叉：[生成式世界模型](../../wiki/methods/generative-world-models.md)、[虚拟沙盒路线](../../wiki/overview/world-models-route-03-virtual-sandbox.md)、[训练闭环 taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)、[Ctrl-World](../../wiki/entities/paper-ctrl-world.md)、[ViTacWorld](../../wiki/entities/paper-vitacworld.md)、[GigaWorld-1](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md)、[Hi-WM](../../wiki/entities/paper-sa-2604-21741-hi-wm-human-in-the-world-model-for-scalable-robo.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Teleoperation](../../wiki/tasks/teleoperation.md)、[舞肌科技](../../wiki/entities/wuji-robotics.md)

## 可信度与使用边界

- 本文为 **公司官方博客**，非同行评审论文；成功率图、真机相关性为 **作者自报**，**独立复现前不宜作硬基准引用**。
- **确认未开源**（截至 2026-08-17）：无训练/推理代码、权重或数据集。
- 架构细节（骨干、分辨率、上下文长度、动作编码）博客未公开；勿与 Ctrl-World / Cosmos 等开源栈默认对齐。
- 跨本体训练的数据配比、各本体小时数未给出。

## Citation

```bibtex
@article{
    currentrobotics2026currentworld0,
    author = {Current Robotics Team},
    title = {CurrentWorld-0: A Cross-Embodiment, Multi-View, Force-Tac World Simulator},
    journal = {Current Robotics Blog},
    year = {2026},
    url = {https://current-robotics.com/blog/currentworld},
}
```
