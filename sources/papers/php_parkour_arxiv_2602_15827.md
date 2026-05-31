# Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching（arXiv:2602.15827）

> 来源归档（ingest）

- **标题：** Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching
- **缩写：** **PHP**
- **类型：** paper / humanoid perceptive locomotion + skill chaining
- **arXiv：** <https://arxiv.org/abs/2602.15827>（HTML：<https://arxiv.org/html/2602.15827v1>）
- **PDF（官方）：** <https://php-parkour.github.io/static/images/paper.pdf>
- **项目页：** <https://php-parkour.github.io/>（浏览器 MuJoCo 演示：<https://php-parkour.github.io/index-mobile.html>）
- **会议：** RSS 2026（OpenReview：<https://openreview.net/forum?id=WzPoEM3McY>）
- **作者：** Zhen Wu*, Xiaoyu Huang*, Lujie Yang*, Yuanhang Zhang, Xi Chen, Pieter Abbeel†, Rocky Duan†, Angjoo Kanazawa†, Carmelo Sferrazza†, Guanya Shi†, C. Karen Liu†（* equal；† Amazon FAR co-lead）
- **机构：** Amazon FAR、UC Berkeley、CMU、Stanford University
- **入库日期：** 2026-05-31
- **一句话说明：** 用 **motion matching** 把稀缺的人类跑酷原子技能与 locomotion 合成为长程运动学参考，再训练多技能 **motion-tracking 专家** 并 **DAgger + PPO** 蒸馏为单一 **深度图学生策略**，在 **Unitree G1** 上仅凭机载深度与离散 2D 速度指令自主完成越障、攀爬、翻越与滚落等长程跑酷。

## 摘要级要点

- **问题：** 人形跑酷需要 (1) 高动态、多接触全身技能；(2) 外感受（视觉）驱动的环境适应；(3) 把大量异质技能 Consolidate 进**单一** visuomotor 策略。动态人类动作数据稀缺（每技能常仅数秒、少量演示），但长程障碍课又需要丰富的**入技能前状态**与**技能间平滑过渡**。
- **核心思路（模块化）：**
  1. **OmniRetarget [43]** 将人类 MoCap 转为机器人可执行**原子跑酷技能库**；
  2. **Motion matching（离线）**：在特征空间最近邻检索，把 locomotion 与原子技能按 `Locomotion → Skill → Locomotion` 合成为多样长程参考（不同接近距离、步态相位、速度档位、地形位姿随机化）；
  3. **专家 RL**：按 BeyondMimic / OmniRetarget 的 motion tracking，特权观测含 0.7 m×0.7 m height scan + 全局根跟踪以纠 drift；
  4. **学生蒸馏**：深度图 + 本体 + 2D 速度指令；**纯 DAgger 不足以学攀爬/翻越**，故 **L = λ_PPO L_PPO + λ_D L_D**（带 warmup curriculum、左右对称终止放宽、均匀技能采样）。
- **感知与接口：** 仅 **机载深度** + **离散 2D 速度命令**（低速 1 m/s / 高速 2 m/s × 五档转向）；策略根据障碍几何**自主选择** step over / climb / vault / roll 等。
- **实机（Unitree G1，1.3 m，29 DoF）：** 最高墙攀 **1.25 m（96% 身高）**、cat vault + dash vault（~3 m/s）、speed vault、48–60 s 多障碍连续穿越、**实时障碍位移**闭环适应；与人体同动作高墙攀对比 timing 接近（机器人 3.63 s toe-off→站稳）。

## 方法细节摘录

### Motion matching 特征与长程合成

- 每帧特征 **x_i**（[14]）：局部坐标系下 (i) 短视界未来轨迹位置与朝向、(ii) 足部关节位置速度、(iii) 根速度。
- 检索：给定当前状态与 2D 速度命令 → 查询特征 **x̂_t** → 在窗口 **C_t** 内 argmin ||x̂_t − x_i||²；每 **M** 帧或速度显著变化时重搜；过渡处短 blending。
- 技能库：每技能手动标注 **(s_k, e_k)** 与入技能窗口 **E_k = [s_k − H_k, s_k]**；技能执行期**不再** motion matching，顺序播放保接触-rich 人类动作；入技能时按参考帧将配对地形对齐到当前根位姿。
- 数据增广：入技能前 locomotion 时长 U[0.1, 3] s（均值 ~0.3 s）；障碍宽/尺寸/ yaw 随机；近轨迹**干扰物** box。

### 训练配置（论文 IV）

- 网络：3 层 CNN + 5 层 MLP [2048, 1024, 512, 256, 128]；**16384** 并行环境；专家与学生各 **20K** iterations。
- 深度：Nvidia **WARP** 渲染；相机外参 ±2.5 cm / ±2.5°；观测延迟 60–80 ms；深度噪声（无高斯 blur 以免高速糊障）。

## 对 wiki 的映射

- 沉淀 / 深化实体页：[`wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md`](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)
- 交叉更新：
  - [DAgger](../../wiki/methods/dagger.md) — PHP 学生蒸馏混合 DAgger + PPO
  - [Locomotion](../../wiki/tasks/locomotion.md) — 感知跑酷 / 技能链
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [OmniRetarget 实体](../../wiki/entities/paper-hrl-stack-03-omniretarget.md) + [omniretarget_arxiv_2509_26633.md](omniretarget_arxiv_2509_26633.md) — 原子技能重定向上游
  - [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md) — 第 22/42 篇

## 关联原始资料

- 项目页归档：[`sources/sites/php-parkour-github-io.md`](../sites/php-parkour-github-io.md)
- 42 篇栈策展（保留）：[`humanoid_rl_stack_22_perceptive_humanoid_parkour_chaining_dynamic_hum.md`](humanoid_rl_stack_22_perceptive_humanoid_parkour_chaining_dynamic_hum.md)
