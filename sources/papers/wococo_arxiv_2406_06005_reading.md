# WoCoCo：通读笔记（arXiv:2406.06005v2 / CoRL 2024 Oral）

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 116/161 + arXiv:2406.06005v2 原文通读与官方仓库源码核对）

- **标题：** WoCoCo: Learning Whole-Body Humanoid Control with Sequential Contacts
- **类型：** paper
- **作者：** Chong Zhang†⋆（ETH Zurich / Work done at CMU）、Tairan He（Carnegie Mellon University）、Wenli Xiao†（Carnegie Mellon University）、Guanya Shi（Carnegie Mellon University）
- **发表/版本：** arXiv:2406.06005v2 [cs.RO]（2024-11-07；CoRL 2024 Oral）
- **Loco-Manip 161 分类：** 05 动捕、人类视频与交互动作规划
- **机构：** ETH Zurich、Carnegie Mellon University
- **项目页：** <https://lecar-lab.github.io/wococo/>
- **论文：** <https://arxiv.org/abs/2406.06005>
- **代码：** <https://github.com/LeCAR-Lab/wococo>
- **核对范围：** 已通读 `arXiv:2406.06005v2` 全文（Sections 1–6 及 Appendix A–L，含 Eqs. 1–11、Supplementary Tables 1–3 与 Figures 7–11）及官方仓库 `LeCAR-Lab/wococo` README 与 `legged_gym`/`rsl_rl` 源码。更正了早期策展摘要中「使用 AMP/运动先验与扩散策略/流匹配」的不实模板描述：原文明确强调 **without any motion priors**，策略为 `[512, 256, 128]` 端到端 MLP，算法为带左右对称数据增强的 PPO，无扩散模型或流匹配组件。
- **通读：** 已完成
- **一句话说明：** WoCoCo 将顺序接触全身控制自然分解为按序完成的离散接触阶段（contact stage = 接触目标 $g_i^{\mathrm{con}}$ + 任务目标 $g_i^{\mathrm{task}}$），用通用奖励 $r_{\mathrm{WoCoCo}} = w_{\mathrm{con}} r_{\mathrm{con}} + w_{\mathrm{stage}} r_{\mathrm{stage}} + w_{\mathrm{curi}} r_{\mathrm{curi}}$ 与三阶段 Sim2Real 课程，仅需每任务 1–2 个直观任务奖励即可训练无需运动先验的端到端 RL 全身控制器。

## 原文核心机制与定量细节（arXiv:2406.06005v2）

### 1. 顺序接触任务分解与扩展 MDP（Section 2, Eq. 1）

- 任务被预先（启发式或由高层接触规划器）分解为接触阶段序列 $i \in \{0, 1, \dots, I\}$；每阶段完成要求同时满足接触目标 $g_i^{\mathrm{con}}$（末端接触状态 $F_{\mathrm{con}}$）与任务目标 $g_i^{\mathrm{task}}$（附加任务要求 $F_{\mathrm{task}}$）。
- 阶段满足后（由接触传感器或人工观测判别），经预设时间延迟切换至下一阶段。
- 总奖励结构：$r = \underbrace{r_{\mathrm{WoCoCo}} + r_{\mathrm{reg}}}_{\text{task-agnostic}} + r_{\mathrm{task}}$（Eq. 1）；端到端 MLP 策略（Actor/Critic 均为 `[512, 256, 128]`，Appendix H.3）输出关节目标位置供低层 PD 跟踪。

### 2. 任务无关 WoCoCo 奖励（Section 3.1, Eqs. 2–6；Appendix A, D, E, I）

- **组合形式：** $r_{\mathrm{WoCoCo}} = w_{\mathrm{con}} r_{\mathrm{con}} + w_{\mathrm{stage}} r_{\mathrm{stage}} + w_{\mathrm{curi}} r_{\mathrm{curi}}$（Eq. 2）。
- **稠密接触奖励（Eq. 3）：** $r_{\mathrm{con}} = n_{\mathrm{corr}} - n_{\mathrm{con}} n_{\mathrm{wrong}} \cdot \mathbf{1}(n_{\mathrm{stage}} > 0) + 2 n_{\mathrm{con}}^2 F_{\mathrm{con}} F_{\mathrm{task}}$。相对简单 0-1 奖励，对每个正确/错误接触精细计分，并在第一阶段（$n_{\mathrm{stage}}=0$）屏蔽错误接触惩罚以防复位即扣分、促进探索（Appendix A 举例解析）。
- **阶段计数奖励（Eq. 4）：** $r_{\mathrm{stage}} = n_{\mathrm{stage}} F_{\mathrm{task}}$，驱动策略突破「停在安全当前阶段」的短视局部最优，其中乘子 $F_{\mathrm{task}}$ 防止为累积阶段奖励而故意不完成当前任务目标。
- **随机网络 Hash 计数好奇心奖励（Eqs. 5–6 & Appendix I）：** 任务无关好奇心观测 $o_{\mathrm{curi}}$（含世界系基座位置/四元数/线角速度，以及所有末端位置与接触状态）经 $[0, \pi]$ 归一化与 $\sin/\cos$ 编码后，送入随机初始化并冻结的单隐层哈希网络 $f_{\mathrm{curi}}$（隐层 `32` 单元、输出 `16` 维，对应 `2^{16}` 桶）：
  - $\mathrm{ID\ of\ bucket}(o_{\mathrm{curi}}) = \mathrm{BIN2DEC}[f_{\mathrm{curi}}(o_{\mathrm{curi}}) > 0]$
  - $r_{\mathrm{curi}} = 1 / \sqrt{\#\mathrm{visits\ of\ bucket}(o_{\mathrm{curi}})}$
  - 并结合对称观测平均（Appendix D）避免 RND 在安全状态之外对高风险稀有姿态过探索。
- **WoCoCo 奖励权重（Supplementary Table 1）：**
  - Parkour：$(w_{\mathrm{con}}, w_{\mathrm{stage}}, w_{\mathrm{curi}}) = (120, 160, 20000)$
  - Loco-Manip.：$(40, 160, 40000)$
  - Dancing：$(10, 5, 5000)$
  - Cliffside：$(20, 40, 10000)$

### 3. 三阶段 Sim2Real 课程与正则化（Section 3.2, Appendix F, H, J, K, L）

- **三阶段训练课程（Section 3.2）：** (1) 无域随机化且开启 $r_{\mathrm{curi}}$ 训练至收敛；(2) 关闭 $r_{\mathrm{curi}}$ 并开启域随机化（Supplementary Table 3：摩擦 $U(0.2,1.1)$、CoM 偏移 $U(-0.1,0.1)\,\mathrm{m}$、连杆质量 $U(0.7,1.3)\times$、Kp/Kd $U(0.75,1.25)\times$、力矩 RFI $0.1\times\tau_{\lim}$、控制延迟 $U(0,20)\,\mathrm{ms}$、每 5 s 随机推扰 $\Delta v_{xy}=0.25\,\mathrm{m/s}$、平地/未见碎石地形）继续训练至收敛；(3) 每 2000 迭代将正则化奖励 $r_{\mathrm{reg}}$ 权重提升 20% 直至翻倍，诱导更平滑保守行为。
- **时序记忆与观测（Section 3.2 & Appendix H）：** 关节位置、关节速度与历史动作堆叠最近 3 个控制步；目标表征由外部动捕原型提供（Parkour 前瞻未来 2 阶段，其余任务仅输入当前阶段以泛化至任意序列长度）。
- **正则化项（Supplementary Table 2）：** 共 11 项通用正则（偏航角速度、力矩、超限力矩、关节加速度/速度、动作变化率、终止惩罚、足端接触力 $>550\,\mathrm{N}$、足端倾角、绊倒与滑移），非跳跃任务额外加 `Feet air time`（权重 20）与 `No fly`（权重 10）。
- **真机控制与 Real-to-Sim 修正（Appendix L）：** 策略 50 Hz、PD 200 Hz，输出经 4 Hz 截止频率 Butterworth 低通滤波防抖动；作者重测并修正官方 H1 URDF 中偏差显著的躯干质量与错误足部几何。

### 4. 实验验证、消融与局限（Sections 4–6, Appendices B, C, G, K）

- **4 项人形任务均完成真机部署（Unitree H1）+ 1 项跨形态仿真：**
  1. **Versatile Parkour Jumping（1 项 $r_{\mathrm{task}}$）：** 单/双脚落石序列 + 上身 `"hug"`/`"relax"` 姿态跟踪，在仿真未见碎石扰动下稳定并真机完成 45 cm 单步跳与连续双脚跳（真机受场地限制仅测双脚跳 1–2 步）。
  2. **Anywhere-to-Anywhere Box Loco-Manipulation（2 项 $r_{\mathrm{task}}$）：** 仅在双手定义 2 个接触阶段（接近并触碰箱侧 -> 保持接触并运送到目标点），零姿势先验下 0.5 s 快速转身、行走-抓取自然过渡、踩中自身绑带后自恢复。
  3. **Dynamic Clap-and-Tap Dancing（2 项 $r_{\mathrm{task}}$）：** `Left`/`Middle`/`Right` 三动作组成序列，要求足端精准落框且在 `Left`/`Right` 拍手碰撞。
  4. **Bidirectional Cliffside Climbing（2 项 $r_{\mathrm{task}}$）：** 四肢精准触达地面/墙面目标框，真机以人手持木板代替悬崖墙，适应手部外力变化。
  5. **22-DoF Dinosaur Ball Loco-Manipulation（仿真）：** 6 个末端（头、尾、四足）轮流推球绘出 `"WoCoCo"` 轨迹。
- **Parkour 消融与稳定/课程验证（Section 5, Figs. 8–11, Appendix K）：**
  - 去除稠密接触奖励（改用 $0$-$1$ 奖励 `c_0-1 * F_con * F_task`）或去除好奇心：完全无法跳上石块，仅原地跟踪上身姿态；
  - 去除阶段计数奖励：故意滞留当前阶段逃避后续风险；
  - 改用 RND 好奇心：学会向后危险仰倒以刷取稀疏生存状态；
  - 5 个随机种子（Fig. 10）：均稳定收敛至任务进度 $> 0.75$（若把 DR 提前到 Phase 1，Parkour 平均进度收敛降至 $< 0.6$；若取消 Phase 3，悬崖攀爬出现急躁抖动并踩出边界，Fig. 11）。
- **局限与未来方向（Section 6）：** (1) 缺少失败预测与安全边界判定；(2) 未知长序列时需启发式裁剪阶段奖励防刷分；(3) 原型阶段依赖外部动捕而非纯机载视觉/LiDAR；(4) 需人工或传感器显式接触反馈来切换阶段，接触序列为启发式预定义。

## 开源与仓库核对

- 对应源码归档见 [lecar_wococo.md](../repos/lecar_wococo.md)。仓库开源于 <https://github.com/LeCAR-Lab/wococo>（CC BY-NC 4.0 + 继承自 Legged Gym / RSL-RL 的协议），公开了完整任务无关组件（`PPO_SYM`、`NHashCuriosity`、`ActionFilterButter`、`OnPolicyRunner`）与 `h1:jumpjack` 拍手舞完整复现示例，其余 3 个人形任务及恐龙任务与论文阶段切换/超参框架一致，按 README 说明鼓励用户依环境自行扩展任务配置。

## 对 wiki 的映射

- [paper-loco-manip-161-116-wococo](../../wiki/entities/paper-loco-manip-161-116-wococo.md)

## 参考来源（原始）

- 论文原文：Zhang et al., *WoCoCo: Learning Whole-Body Humanoid Control with Sequential Contacts*, arXiv:2406.06005v2, 2024. <https://arxiv.org/abs/2406.06005>
- 项目主页：<https://lecar-lab.github.io/wococo/>
- 官方代码库：<https://github.com/LeCAR-Lab/wococo>（归档：[lecar_wococo.md](../repos/lecar_wococo.md)）
- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
