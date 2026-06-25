# MoRE: Mixture of Residual Experts for Humanoid Lifelike Gaits Learning on Complex Terrains（arXiv:2506.08840）

> 来源归档（ingest）

- **标题：** MoRE: Mixture of Residual Experts for Humanoid Lifelike Gaits Learning on Complex Terrains
- **类型：** paper / humanoid locomotion / AMP / mixture-of-experts / perceptive locomotion / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2506.08840>
- **arXiv HTML：** <https://arxiv.org/html/2506.08840v1>
- **PDF：** <https://arxiv.org/pdf/2506.08840>
- **项目页：** <https://more-humanoid.github.io/>
- **机构：** 中国科学技术大学（USTC）、中国电信人工智能研究院（TeleAI）、哈尔滨工程大学（HEU）、上海科技大学（ShanghaiTech）；通讯作者 Chenjia Bai
- **硬件：** Unitree G1 + Intel RealSense D435i 深度相机
- **入库日期：** 2026-06-25
- **一句话说明：** 两阶段训练：先用深度相机学复杂地形 base locomotion，再以 **latent residual MoE + 多判别器 AMP** 在 gait command 下切换 Walk-Run / High-Knees / Squat 等人形步态，G1 真机 50 Hz 部署。

## 摘要级要点

- **问题：** 纯本体感知或平地 AMP 难以在 **台阶/沟壑/高台** 等复杂地形上保持 **可切换的人形步态**；MoCap 参考多来自平地，单参考 AMP 也难覆盖高动态平衡动作。
- **两阶段管线：**
  1. **Stage 1 — Base locomotion：** 随机初始化 actor-critic，**仅用 locomotion 奖励 $\bm{r}^l$**，输入含 **头部深度图** + 本体历史；critic 用特权高程图；无 motion prior、无 gait command。学完后可过楼梯、沟壑、高台与坡地。
  2. **Stage 2 — Anthropomorphic gaits：** 冻结 base actor 主干，挂载 **Mixture of Latent Residual Experts（MoRE）**；残差 $\bm{z}'_t$ 加到 actor 末层隐特征 $\bm{z}^o_t$ 上再进 action head；输入含 **one-hot gait command** $\bm{c}^g_t$；**多判别器 AMP**（每步态一个）+ **gait-specific 奖励** $\bm{r}^g$ 联合优化。
- **多判别器 AMP：** 每步态 $i$ 独立判别器 $D_{\phi_i}$；风格奖励按 gait command 只从对应判别器取；参考为 LAFAN1 retarget 到 G1 的 **5 步关节角轨迹** $\tau$（非单步转移）。
- **MoE 残差：** $N$ 个 expert MLP + gate 网络对 expert 输出加权求和，缓解多技能梯度冲突；论文实验取 **3 个 expert**。
- **三种步态（gait command）：** Walk-Run、High-Knees、Squat（蹲走）；gait rewards 约束基座高度、抬膝高度等，使风格不必完全复制参考动作。
- **感知：** 非对称 actor-critic；actor 用 64×64 深度图（两帧）+ 本体；仿真用 NVIDIA Warp 渲染深度；域随机化含深度噪声、相机位姿、自遮挡模拟。
- **训练：** Isaac Gym；Stage 1 ~10k iter（1×RTX 4090）；Stage 2 ~20k iter（4×RTX 4090）；地形课程：gap 0.05–0.45 m、step 0.05–0.3 m、stair 0.05–0.15 m。
- **动作：** 16 维关节目标（肩 pitch、肘 pitch、全腿）；PD 底层控制。
- **真机：** TCP 多进程；D435i 640×480 → 滤波 → 64×64；相机 10 Hz、策略 50 Hz。

## 核心摘录（面向 wiki 编译）

### 与相邻 AMP / 感知 locomotion 路线对照

| 维度 | MoRE（本文） | 平地单参考 AMP | SD-AMP（#10） | Hiking in the Wild（#09） |
|------|-------------|----------------|---------------|---------------------------|
| 外感知 | **深度相机** | 通常仅本体 | 仅本体 | 深度 + 跑酷框架 |
| 先验形态 | **多判别器 + gait command** | 单判别器 | **双判别器 + 重力门控** | 非典型 AMP |
| 步态切换 | **显式 gait command + MoE 残差** | 单一风格 | 速度条件 walk/run + recovery | 感知落脚为主 |
| 地形目标 | 楼梯/沟壑/台阶组合 | 平地/缓坡 | 走跑起身统一 | 野外跑酷 |

### 仿真 benchmark（Table III 摘要，成功率 Succ.）

相对 **Blind Locomotion** 与 **Stage-1 Base Locomotion**，MoRE 在 8 m×14 m 标准赛道（gap / stair / step，Easy+Hard）上三类步态均显著提升；例：Hard Stair Walk-Run **0.682 vs Base 0.660 vs Blind 0.218**；Hard Gap High-Knees **0.904 vs Base 0.893 vs Blind 0.517**。

### 策展导读（AMP 专题 #08）

- 自然步态不是单一风格，而是 **地形与任务条件下的可切换运动模式**；MoRE 用多 expert + 多判别器让「像人」具备 **状态/命令依赖**。
- AMP 在人形上不能只做平地美化，必须进入 **地形变化 + 步态切换**（见 [wechat_humanoid_amp_19_survey](../raw/wechat_humanoid_amp_19_survey_2026-05-26.md) 论文 08 段）。

## 对 wiki 的映射

- 沉淀实体页：[MoRE（AMP 专题 #08）](../../wiki/entities/paper-amp-survey-08-more.md)
- 交叉补强：[AMP & HumanX](../../wiki/methods/amp-reward.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Terrain Adaptation](../../wiki/concepts/terrain-adaptation.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[LAFAN1](../../wiki/entities/lafan1-dataset.md)、[显式楼梯几何条件化](../../wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md)（MoRE 作视觉基线）、[Hiking in the Wild](../../wiki/entities/paper-hiking-in-the-wild.md)、[ALMI #07](../../wiki/entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md)

## 参考来源（原始）

- arXiv:2506.08840 — 论文正文
- [humanoid_amp_survey_08_more_mixture_of_residual_experts_for_humanoid_li.md](humanoid_amp_survey_08_more_mixture_of_residual_experts_for_humanoid_li.md) — AMP 19 篇策展索引
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md) — 微信公众号编译导读
