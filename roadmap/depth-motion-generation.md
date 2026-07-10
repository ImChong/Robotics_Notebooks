# 路线（纵深）：如果目标是动作生成（文本/多模态 → 人形动作）

**摘要**：面向"想用生成模型造出人体/人形动作"的纵深路线，从生成模型基础（扩散 / 流匹配）与动作表示、文本-动作数据集，到生成骨干主线（MDM 式扩散 → 可控生成 → HY-Motion / GENMO / Kimodo 规模化三线），再到物理化落地（PhysDiff / PhyGile / 重定向桥）与控制环内生成器（扩散规划器 / OMG / MotionBricks / GPC），按 Stage 0–5 串通核心方法；本路线是 [运动控制主路线](motion-control.md) 的一条分支，与 [动作重定向纵深](depth-motion-retargeting.md) 构成"造动作 / 落到机器人"的姊妹路线。

## 路线一览

```mermaid
flowchart LR
  S0["**Stage 0**<br/>全景与前置<br/><em>扩散 / 流匹配基础</em>"]
  S1["**Stage 1**<br/>表示与数据<br/><em>SMPL 系 / HumanML3D → MotionMillion</em>"]
  S2["**Stage 2**<br/>生成骨干主线<br/><em>MDM → 可控生成 → 规模化三线</em>"]
  S3["**Stage 3**<br/>物理化落地<br/><em>PhysDiff / PhyGile / 重定向桥</em>"]
  S4["**Stage 4**<br/>控制环内生成器<br/><em>扩散规划 / OMG / GPC</em>"]
  S5["**Stage 5**<br/>进阶方向<br/><em>视频生成 / 零样本 / 场景交互</em>"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5

  classDef stage fill:#142a3a,stroke:#fd79a8,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4,S5 stage
```

## 这条路径怎么用

- 目标读者是有深度学习基础、想让"一段文字/视频/约束变成一段动作"的人——既包括人体动作合成，也包括为人形机器人生产参考动作与在线规划
- 动作生成解决 **参考动作从哪来**：动捕录不全、成本高，生成模型把参考动作的供给从"录多少用多少"变成"要什么造什么"；但要牢记硬约束——**生成输出是运动学层，不等于物理可行的机器人指令**
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 本路线是主路线 L5（RL 与模仿学习）之后偏"数据与规划侧"的进阶方向，生成产物经 [动作重定向纵深](depth-motion-retargeting.md) 与跟踪器（SONIC / BeyondMimic 一系）才能落到真机
- 与 [BFM 纵深](depth-bfm.md) 在"动作先验"上互补：BFM 把行为先验压进策略权重，动作生成把行为先验放在显式的运动学轨迹层
- 如果最终目标是"语言指令 → 人形整机执行"，本路线 Stage 3–4 与 [VLA 纵深](depth-vla.md) 的分层接口设计相互印证

---

## Stage 0 生成模型基础与动作生成全景

**先把扩散与流匹配的数学骨架立起来，再进方法谱系，否则每篇论文的"新采样器"都会看成新范式。**

### 前置知识
- Python + PyTorch 熟练
- 理解概率论基础与神经网络训练

### 核心问题
- 扩散模型的去噪过程与训练目标是什么，为什么适合多峰分布的动作数据
- 流匹配（flow matching）与扩散的关系：确定性 ODE 采样换来什么
- 动作生成在机器人栈里的位置：离线数据引擎 vs 在线参考规划器

### 推荐做什么
- 在 2D 玩具分布上手写一个最小扩散模型，观察去噪轨迹
- 读 probability flow 形式化页，推一遍"SDE → 概率流 ODE"的等价关系

### 推荐读什么
- [Diffusion Model](../wiki/concepts/diffusion-model.md)（本仓库）— 生成基础主入口
- [Probability Flow](../wiki/formalizations/probability-flow.md)（本仓库）— 流匹配的数学骨架
- [Diffusion-based Motion Generation](../wiki/methods/diffusion-motion-generation.md)（本仓库）— 动作生成总览页
- [Awesome Text-to-Motion](../wiki/entities/awesome-text-to-motion-zilize.md)（本仓库）— 领域论文地图

### 学完输出什么
- 能推导扩散训练目标与采样过程
- 能说清"离线数据引擎"与"在线规划器"两种用法的技术要求差异

---

## Stage 1 动作表示与数据集

**表示定上限：SMPL 系参数、关节旋转、接触标签的每一个选择都会传导到生成质量与下游可用性。**

### 前置知识
- Stage 0 内容
- 了解人体骨架与旋转表示（可参考 [动作重定向纵深](depth-motion-retargeting.md) Stage 1）

### 核心问题
- SMPL / SMPL-H / SMPL-X 表示族的差异，为什么成为文本-动作数据的事实标准
- HumanML3D 263 维表示包含什么、为什么 HY-Motion 刻意弃用它
- 文本-动作数据集谱系：KIT-ML → HumanML3D → BABEL → Motion-X → MotionMillion 的规模与标注方式演进
- 脚部接触标签、根轨迹表示对生成质量的影响

### 推荐做什么
- 加载 HumanML3D 一条样本，把 263 维表示逐段解析成根轨迹 / 关节位置 / 接触标签
- 对比两个数据集的文本标注风格，思考"改写 LLM"为什么成为生成管线标配

### 推荐读什么
- [HumanML3D](../wiki/entities/dataset-bfm-humanml3d.md)、[KIT-ML](../wiki/entities/dataset-bfm-kit-ml.md)、[BABEL](../wiki/entities/dataset-bfm-babel.md)（本仓库）— 文本-动作数据基座
- [Motion-X](../wiki/entities/dataset-bfm-motion-x.md) 与 [PoseScript](../wiki/entities/dataset-bfm-posescript.md)（本仓库）— 规模与细粒度标注扩展
- [Go to Zero / MotionMillion](../wiki/entities/paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w.md)（本仓库）— 百万级数据与零样本评测
- [Human Motion 论文分类页](../wiki/overview/paper-notebook-category-14-human-motion.md)（本仓库）

### 学完输出什么
- 能读写 SMPL 系动作数据并解析主流表示格式
- 一份数据集选型表（规模 / 标注 / 许可 / 重定向就绪度）

---

## Stage 2 生成骨干主线：从 MDM 到规模化三线

**主线一条：扩散把文本-动作生成带进主流，可控性与规模化把它推向工程可用。**

### 前置知识
- Stage 1 内容

### 核心问题
- MDM 式动作扩散怎么把去噪过程用在骨架序列上
- 可控生成的机制族：GMD 的引导式扩散、OmniControl 的任意关节时空约束、in-betweening 的关键帧补全
- 规模化三线的取舍：HY-Motion（十亿级流匹配 DiT + DPO/Flow-GRPO 对齐）、GENMO（估计与生成 dual-mode 统一）、Kimodo（两阶段 root/body 去噪 + 强约束可控）
- 对齐与后训练（DPO / Flow-GRPO）在动作生成里解决什么问题

### 推荐做什么
- 跑通一个开源文本-动作模型（HY-Motion 或 MDM 系）的推理 demo，观察同一 prompt 的多样性
- 精读三线对比页，为"数据引擎 / 视频驱动 / 导演式编辑"三种场景各选一条骨干并说明理由

### 推荐读什么
- [HY-Motion 1.0](../wiki/methods/hy-motion-1.md)、[GENMO](../wiki/methods/genmo.md)、[Kimodo](../wiki/entities/kimodo.md)（本仓库）— 规模化三线
- [HY-Motion vs GENMO vs Kimodo 选型对比](../wiki/comparisons/hy-motion-vs-genmo-vs-kimodo.md)（本仓库）— 谱系主入口
- [Guided Motion Diffusion](../wiki/entities/paper-notebook-guided-motion-diffusion-for-controllable-human-m.md) 与 [OmniControl](../wiki/entities/paper-notebook-omnicontrol-control-any-joint-at-any-time-for-hu.md)（本仓库）— 可控生成机制
- [Flexible Motion In-betweening](../wiki/entities/paper-notebook-flexible-motion-in-betweening-with-diffusion-mod.md)（本仓库）— 关键帧补全

### 学完输出什么
- 一个跑通的文本-动作生成 demo 与生成质量的定性判断标准
- 能说清三条规模化路线"吃什么条件、押什么叙事、差在哪"

---

## Stage 3 物理化落地：从运动学输出到机器人可执行

**生成的动作会飘、会滑、会穿模：物理化是动作生成进机器人栈的必修课。**

### 前置知识
- Stage 2 内容
- [动作重定向纵深](depth-motion-retargeting.md) Stage 0–3 水平（知道重定向解什么）

### 核心问题
- 生成输出的物理伪影从哪来：训练数据是运动学层、损失里没有动力学
- 三条物理化路线：物理引导采样（PhysDiff）、robot-native 生成 + 跟踪器共训（PhyGile）、生成 → 重定向 → 跟踪三段桥（Gen2Humanoid）
- 生成产物进入跟踪训练的链路与失败模式
- 人体先验 retarget 与机器人骨骼空间直接生成的取舍

### 推荐做什么
- 用 Gen2Humanoid 把一句文本 prompt 变成人形关节轨迹，在 viser 里检查重定向伪影
- 把生成轨迹喂给一个开源跟踪器（SONIC / BeyondMimic 系），统计可跟踪比例

### 推荐读什么
- [PhysDiff](../wiki/entities/paper-notebook-physdiff-physics-guided-human-motion-diffusion-m.md)（本仓库）— 物理引导扩散
- [PhyGile](../wiki/entities/paper-phygile.md)（本仓库）— robot-native 生成 + GMT 闭环
- [Gen2Humanoid](../wiki/entities/gen2humanoid.md)（本仓库）— 生成 → 重定向端到端管线
- [SONIC](../wiki/methods/sonic-motion-tracking.md) 与 [Whole-Body Tracking Pipeline](../wiki/concepts/whole-body-tracking-pipeline.md)（本仓库）— 跟踪侧消费者

### 学完输出什么
- 一条"文本 → 生成 → 重定向 → 跟踪验证"的端到端管线
- 对三条物理化路线的适用场景有基于实验的判断

---

## Stage 4 控制环内的生成器：从数据引擎到在线规划

**离线造数据只是入口：把生成器放进控制回路，才是当前人形系统的前沿形态。**

### 前置知识
- Stage 3 内容
- [RL 纵深路线](depth-rl-locomotion.md) Stage 0–2 水平（能在仿真里训练跟踪策略）

### 核心问题
- 生成器进控制环的分布偏移问题：闭环微调（closed-loop fine-tuning）为什么必要
- 推理延迟怎么压：少步去噪、TensorRT、receding-horizon 异步重规划
- 生成式中间件的位置选择：地形条件规划器（ETH G1）、恢复中间层（Heracles）、多模态接口（OMG）、实时身体 API（MotionBricks）
- token 化生成控制器（GPC）与轨迹级扩散生成的范式差异

### 推荐做什么
- 精读 ETH G1 三阶段耦合（数据增广 → 离线预训练 → 冻结生成器闭环微调），画出数据流图
- 对比"固定参考跟踪"与"在线生成参考"在 OOD 场景的成功率报告，理解在线生成换来什么

### 推荐读什么
- [Diffusion-based Motion Generation](../wiki/methods/diffusion-motion-generation.md)（本仓库）— 控制环内生成器总览
- [Learning Whole-Body Humanoid Locomotion](../wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) 与 [Heracles](../wiki/entities/paper-heracles-humanoid-diffusion.md)（本仓库）— 扩散规划与恢复中间件
- [OMG](../wiki/entities/paper-omg-omni-modal-humanoid-control.md) 与 [MotionBricks](../wiki/methods/motionbricks.md)（本仓库）— 多模态接口与实时基元
- [GPC](../wiki/entities/paper-gpc-generative-pretrained-controllers.md)（本仓库）— token 化生成控制器

### 学完输出什么
- 能为"生成器 + 跟踪器"耦合系统设计训练与部署方案（含延迟预算）
- 对"生成器放在栈的哪一层"有基于任务的判断

---

## Stage 5 进阶方向

### 前置知识
- Stage 4 内容

**方向 A：视频生成 → 机器人轨迹**
- 用视频生成模型的动作先验反哺机器人：生成人类动作视频，再估计并物理化
- 关键词：[From Generated Human Videos to Physically Plausible Robot Trajectories](../wiki/entities/paper-hrl-stack-04-from_generated_human_videos_to_physi.md)、[Mimic-Video](../wiki/methods/mimic-video.md)

**方向 B：零样本与 Scaling**
- 百万级数据 + 十亿级参数把文本-动作推向零样本泛化
- 关键词：[Go to Zero / MotionMillion](../wiki/entities/paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w.md)、[GPC](../wiki/entities/paper-gpc-generative-pretrained-controllers.md)、[HY-Motion 1.0](../wiki/methods/hy-motion-1.md)

**方向 C：动作生成 × 行为基础模型**
- 生成器为 BFM 供给行为先验、BFM 为生成器提供物理执行底座
- 关键词：[Kimodo](../wiki/entities/kimodo.md)（直出 G1 / ProtoMotions 闭环）、[BFM 纵深路线](depth-bfm.md)

**方向 D：场景与交互感知生成**
- 从"空场地生成"走向"贴着场景与物体生成"
- 关键词：[DIMOS](../wiki/entities/paper-dimos-human-scene-motion-synthesis.md)、[Loco-Manipulation](../wiki/tasks/loco-manipulation.md)、[移动操作纵深路线](depth-mobile-manipulation.md)

---

## 快速入口汇总

| 阶段 | 核心问题 | 本仓库入口 |
|------|---------|-----------|
| Stage 0 | 扩散 / 流匹配基础 | [Diffusion Model](../wiki/concepts/diffusion-model.md) |
| Stage 1 | 表示与数据集 | [HumanML3D](../wiki/entities/dataset-bfm-humanml3d.md) |
| Stage 2 | 生成骨干选型 | [HY-Motion vs GENMO vs Kimodo](../wiki/comparisons/hy-motion-vs-genmo-vs-kimodo.md) |
| Stage 3 | 物理化落地 | [Gen2Humanoid](../wiki/entities/gen2humanoid.md) |
| Stage 4 | 控制环内生成器 | [Diffusion-based Motion Generation](../wiki/methods/diffusion-motion-generation.md) |
| Stage 5 | 进阶方向 | [Awesome Text-to-Motion](../wiki/entities/awesome-text-to-motion-zilize.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 其它纵深路径：
  - [动作重定向（人体动作 → 机器人参考轨迹）](depth-motion-retargeting.md) — 姊妹路线：生成负责"造动作"，重定向负责"落到机器人"
  - [BFM（人形行为基础模型）](depth-bfm.md) — 行为先验的策略权重侧表达
  - [模仿学习与技能迁移](depth-imitation-learning.md) — Diffusion Policy 一系与本路线共享生成式建模基础
  - [VLA（视觉-语言-动作模型）](depth-vla.md) — 语义接口与分层设计的邻接路线
  - [人形 RL 运动控制](depth-rl-locomotion.md) — 跟踪器训练的训练侧前置
  - [移动操作（Loco-Manipulation）](depth-mobile-manipulation.md) — Stage 5 方向 D 的展开版
  - [传统模型控制（LIP/ZMP → MPC → WBC）](depth-classical-control.md)
  - [安全控制（CLF/CBF）](depth-safe-control.md)
  - [接触丰富的操作任务](depth-contact-manipulation.md)
  - [导航（SLAM → VLN → 导航 VLA）](depth-navigation.md)
  - [感知越障（Perceptive Locomotion）](depth-perceptive-locomotion.md)
- 人形控制全景图：[Humanoid Control Roadmap](../wiki/roadmaps/humanoid-control-roadmap.md)
- 技术栈地图：[tech-map/dependency-graph.md](../tech-map/dependency-graph.md)

## 参考来源

本路线基于以下原始资料的归纳：

- [Diffusion-based Motion Generation](../wiki/methods/diffusion-motion-generation.md) 与 [HY-Motion vs GENMO vs Kimodo 选型对比](../wiki/comparisons/hy-motion-vs-genmo-vs-kimodo.md)
- "Human Motion Diffusion Model" (Tevet et al., 2022, MDM) — 文本驱动动作扩散生成范式的确立
- "HY-Motion 1.0" (Tencent Hunyuan, arXiv:2512.23464) — 流匹配 DiT 规模化代表
- "Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking" (ETH, arXiv:2604.17335) — 控制环内扩散生成器代表
