---
type: method
tags: [locomotion, diffusion, generative-model, humanoid]
status: complete
updated: 2026-06-19
related:
  - ../entities/kimodo.md
summary: "利用扩散模型生成机器人全身运动序列，通过闭环微调解决分布偏移，实现复杂地形下的实时运动规划。"
---

# Diffusion-based Motion Generation (基于扩散模型的运动生成)

**Diffusion-based Motion Generation**：利用扩散概率模型（Diffusion Probabilistic Models）生成机器人关节空间或笛卡尔空间的连续运动序列，通常作为分层控制架构中的高层参考规划器。

## 一句话定义

利用扩散模型的去噪过程预测未来一段时间内的全身运动轨迹，为底层控制器提供具备地形感知能力的高质量参考动作。

## 核心原理

基于扩散的运动生成将运动预测视为一个去噪过程。给定当前状态 $s_t$ 和条件项 $c$（如地形信息、导航目标），模型学习从随机噪声中恢复出一段未来运动轨迹 $\tau = \{q_{t+1}, \dots, q_{t+H}\}$。

### 主要特点
1. **多峰分布建模**：能够捕获人类运动的多样性，解决传统确定性模型在复杂决策点的“均值平滑”问题。
2. **长程一致性**：相比于逐帧预测，轨迹生成能够保证运动在时间窗口内的物理连贯性。
3. **条件约束**：可以轻松整合地形图（Elevation Maps）、任务目标或文本指令作为生成条件。

## 控制环内的生成式中间件（Heracles）

[Heracles](../entities/paper-heracles-humanoid-diffusion.md)（arXiv:2603.27756）将 **conditional flow matching** 用作 **tracker 与原始参考之间的中间层**：状态贴近参考时残差≈0（近似恒等映射），大偏差时生成短视界恢复关键帧并 **receding-horizon 重规划** 后交给底层 RL tracker——区别于下文 ETH G1 **地形条件扩散规划 + RL 跟踪** 分层，但同属 **生成参考 + 物理执行** 家族。

## 文本→机器人原生生成 + GMT 闭环（PhyGile，arXiv:2603.19305）

[PhyGile](../entities/paper-phygile.md) 针对 **人体 text-to-motion 重定向** 的物理不可行问题：用 **TP-MoE 条件扩散** 在 **262D 机器人骨骼空间** 从文本直接生成运动，以 **physics-derived prefix** 引导推理续写，并经 **预训练 GMT** 验证、**闭环仿真精炼** 与 prefix 阶段 **GMT 微调** 闭合生成–执行环。真机展示 breakdance、侧手翻、高踢与空中旋跳等 **高动态全身** 动作。与 [Harmon](../entities/paper-notebook-harmon.md) 同属语言驱动人形生成，但 PhyGile 强调 **robot-native 空间** 与 **跟踪器共训** 而非人体先验 retarget。

## Omni-modal 人形运动生成（OMG，清华 MARS Lab）

[OMG](../entities/paper-omg-omni-modal-humanoid-control.md) 将 **OMG-DiT** 扩散 Transformer 作为 **generator 层**，把 **语言 / 音频 / 人体参考 / 运动历史及零样本组合** 实时转为 **Unitree G1** 可跟踪全身轨迹，再由 [HoloMotion](../entities/holomotion.md) tracker 执行；配套 **OMG-Data**（约 1174.66 h，sim-in-the-loop 过滤）与 **50M–1B** scaling。与 ETH G1 的 **地形条件规划**、Heracles 的 **recovery 中间件** 对照，OMG 强调 **多模态控制接口 + foundation 行为** 与 **运行时 modality switching**。

## 真机全身感知 locomotion（ETH G1，arXiv:2604.17335）

[Learning Whole-Body Humanoid Locomotion](../entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) 将 MDM 式扩散生成器与 DeepMimic 式 RL 跟踪器 **三阶段** 耦合（数据增广 → 离线预训练 → **冻结生成器的闭环微调**），在 Unitree G1 上实现 onboard LiDAR 地形感知下的箱攀、跨栏、楼梯与混合地形穿越；部署用 TensorRT 2 步去噪（~20 ms）+ 0.25 s receding-horizon 参考更新。量化上，**在线生成** 相对固定参考跟踪显著提升 OOD 障碍高度/偏航泛化，**闭环微调** 进一步提升耦合系统成功率。

## RL + 生成式运动基元潜空间（DIMOS，ICCV 2023）

与扩散去噪不同，[DIMOS](../entities/paper-dimos-human-scene-motion-synthesis.md) 把 **CVAE 运动基元的潜变量** 当作 RL 动作，在 **室内 3D 场景** 上学 **场景感知导航 + marker 引导坐/躺/站起**；属于 **生成先验 + 任务策略** 家族，但面向 **运动学 SMPL-X 角色填充** 而非机器人地形规划。与本文 ETH G1 / Heracles 路线的对照：DIMOS 无扩散迭代延迟，强项是 **人–家具交互序列** 与 **无需人体–场景配对轨迹**。

## 主要技术路线 (以 ETH G1 为例)

1. **输入表示**：
   - 历史本体感受状态（Proprioception History）。
   - 目标指令（Yaw velocity, Heading）。
   - 局部地形扫描（Local Elevation Map）。
2. **扩散生成架构**：
   - 采用 1D CNN 或 Transformer 结构的骨干网。
   - 预测未来约 0.5s - 1.0s 的全身参考姿态。
3. **闭环微调（Closed-loop Fine-tuning）**：
   - 在仿真环境中，使底层 RL 策略在扩散生成器的“实时指导”下进行演练，学习适应生成器的噪声，解决分布偏移（Distribution Mismatch）问题。

## 关键挑战与解决方案

### 1. 分布偏移
- **问题**：扩散模型在离线数据上训练，但底层跟踪器（Tracking Controller）在执行时产生的细微偏差会导致生成器进入未见过的状态空间。
- **方案**：闭环微调，将生成器集成进仿真训练循环。

### 2. 推理延迟
- **问题**：扩散模型迭代次数多，计算量大。
- **方案**：
  - **收缩时界（Receding-horizon）更新**：异步触发推理。
  - **加速库**：使用 NVIDIA TensorRT 进行模型量化与推理加速。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| CNN | Convolutional Neural Network | 卷积神经网络，处理图像/深度感知 |
| SMPL | Skinned Multi-Person Linear Model | 常见人体参数化模型与重定向源 |
| DiT | Diffusion Transformer | 以 Transformer 为骨干的扩散生成架构 |
| HOI | Human–Object Interaction | 人与物体接触交互的技能场景 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |

## 参考来源
- [机器人论文阅读笔记：Learned Motion Matching](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Learned_Motion_Matching/Learned_Motion_Matching.html)
- [OMG 项目页与仓库](../../sources/sites/omg-tsinghua-mars-lab-github-io.md) — 清华 MARS Lab omni-modal G1 运动生成（OMG-DiT + HoloMotion tracker）
- [Heracles（arXiv:2603.27756）](../../sources/papers/heracles_humanoid_diffusion_arxiv_2603_27756.md) — 状态条件 flow matching 中间件 + 闭环 tracker（真机恢复）
- [sources/papers/eth-g1-diffusion.md](../../sources/papers/eth-g1-diffusion.md) — ETH Zurich 2026 G1 扩散运动生成工作，结合扩散模型与 RL 跟踪器实现全身移动。
- [sources/repos/kimodo.md](../../sources/repos/kimodo.md) — Kimodo：大规模动捕上训练的运动扩散模型与约束式生成工具链（SOMA / G1 / SMPL-X）。
- [sources/papers/kimodo_arxiv_2603_15546.md](../../sources/papers/kimodo_arxiv_2603_15546.md) — Kimodo（arXiv:2603.15546）：两阶段 root/body 运动学扩散与 Rigplay 700h scaling。
- [sources/sites/kimodo-project.md](../../sources/sites/kimodo-project.md) — NVIDIA 官方项目页：能力演示、生态互操作与机器人应用叙事。
- [sources/papers/genmo.md](../../sources/papers/genmo.md) — GENMO（ICCV 2025 Highlight，NVIDIA）：把人体运动估计形式化为带观测约束的扩散生成，dual-mode 训练统一估计 + 生成。
- [sources/papers/hy_motion_arxiv_2512_23464.md](../../sources/papers/hy_motion_arxiv_2512_23464.md) — HY-Motion 1.0（腾讯混元，arXiv:2512.23464）：十亿级 DiT+流匹配的文本→SMPL-H 运动与全阶段对齐管线。
- [sources/papers/dart_control_arxiv_2410_05260.md](../../sources/papers/dart_control_arxiv_2410_05260.md) — DART / DartControl（ICLR 2025，arXiv:2410.05260）：自回归运动原语潜扩散 + 在线文本流与潜空间空间控制。
- [sources/repos/zilize-awesome-text-to-motion.md](../../sources/repos/zilize-awesome-text-to-motion.md) — Zilize 维护的文本驱动人体运动生成综述/数据集/模型精选与交互式项目页索引。
- [PhyGile（arXiv:2603.19305）](../../sources/papers/phygile_arxiv_2603_19305.md) — physics-prefix 引导的 262D 机器人原生扩散 + GMT 闭环（真机高动态）
- [Diffusion Policy](./diffusion-policy.md) — 扩散策略在操作任务中的应用。
- [GENMO（统一人体运动估计与生成）](./genmo.md) — 人体运动域的扩散生成代表实现，与机器人控制域的扩散运动生成相互参照（估计 ↔ 生成的双向收益）。

## 关联页面
- [Learning Whole-Body Humanoid Locomotion（ETH G1）](../entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) — 地形条件扩散规划 + RL 全身跟踪，真机验证
- [OMG](../entities/paper-omg-omni-modal-humanoid-control.md) — omni-modal 生成器 + HoloMotion tracker，G1 真机多模态切换
- [Heracles](../entities/paper-heracles-humanoid-diffusion.md) — 控制环内生成式中间件（flow matching + tracking）
- [PhyGile](../entities/paper-phygile.md) — 文本→262D robot-native 扩散 + GMT 生成–跟踪闭环
- [DIMOS](../entities/paper-dimos-human-scene-motion-synthesis.md) — RL + CVAE 潜空间，室内人–场景交互运动合成（非扩散）
- [HY-Motion 1.0](./hy-motion-1.md) — 十亿级 DiT+流匹配的人体文本→运动开源系列（腾讯混元）
- [DART（DartControl）](./dart-control.md) — 自回归原语潜扩散 + 在线文本与空间控制（ICLR 2025，ETH）
- [Kimodo（实体页）](../entities/kimodo.md) — 文本 + 运动学约束的人形/人体运动扩散官方实现
- [Awesome Text-to-Motion（Zilize）](../entities/awesome-text-to-motion-zilize.md) — 人体文本–运动文献与数据集的 curated 入口（单人、无 HOI）
- [Humanoid Locomotion](../tasks/humanoid-locomotion.md)
- [Motion Retargeting](../methods/motion-retargeting-gmr.md)
- [PPO](./policy-optimization.md)
- [Probability Flow](../formalizations/probability-flow.md) — 扩散模型的数学基础
- [Contact Dynamics](../concepts/contact-dynamics.md)
