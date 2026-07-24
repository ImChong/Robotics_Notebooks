# Unleashing the Agility of Wheeled-Legged Robots for High-Dynamic Reflexive Obstacle Evasion（arXiv:2604.23761）

> 来源归档（ingest）

- **标题：** Unleashing the Agility of Wheeled-Legged Robots for High-Dynamic Reflexive Obstacle Evasion
- **缩写 / 框架：** **AWARE**（Adaptive Wheeled-Legged Avoidance and Reflexive Evasion）
- **类型：** paper / wheeled-legged / reflexive evasion / hierarchical RL / dual-mode locomotion / dynamic obstacle avoidance / sim2real
- **arXiv：** <https://arxiv.org/abs/2604.23761>（HTML：<https://arxiv.org/html/2604.23761>；PDF：<https://arxiv.org/pdf/2604.23761>）
- **项目页：** 无独立项目页（截至入库日；作者主页 <https://cehao1.github.io/> 亦未列 AWARE 代码入口）
- **代码：** arXiv abs/HTML/API/PDF 均未列官方 GitHub / 权重；**确认未开源**（截至 2026-07-24）
- **作者：** Yongen Zhao、Zihao Xu、Wenzhi Lu、Zhen Chu、Ce Hao
- **机构：** 天津大学机械工程学院（TJU）；新加坡国立大学计算学院（NUS）；北京中关村学院（Zhongguancun Academy）；云深处科技（Deep Robotics，杭州）
- **硬件 / 仿真：** Deep Robotics **M20** 轮足平台；仿真 **NVIDIA Isaac Lab**；真机用动捕提供机器人与障碍位置/速度
- **入库日期：** 2026-07-24
- **一句话说明：** 面向轮足机器人的 **高动态反射式避障**：用分层 RL 把威胁感知决策与 **双模态低层专家**（低速全向导航避让 / 高动态敏捷反射）解耦，涌现前冲、侧闪等混合步态，并在 Isaac Lab 与 M20 真机（抛箱、棍戳、踢腿）上验证。

## 开源状态（步骤 2.5）

- **项目页：** 无 `*.github.io` / lab 资源页；公开入口仅为 arXiv（8 pages, 8 figures, 4 tables）。
- **代码核查（2026-07-24）：** arXiv API 无 comment 声明 code；HTML 全文无 GitHub / Hugging Face 链接；Ce Hao 个人主页 Research Projects 未列 AWARE 仓库。
- **结论：** **确认未开源**。复现需自行按正文在 Isaac Lab 实现双专家 + 高层 Gumbel-Softmax 模式选择与域随机化表。

## 摘录 1：问题与主张（Abstract / §I）

- **痛点：** 轮足兼顾滚动能效与足式越障，但在人机混杂环境中需对快速动态障碍做 **反射式规避**；混合形态、模态耦合与非完整约束使高速全身规避难。
- **对照：** 传统 NMPC/WBC 在高维全身上算力贵；既有敏捷避障多聚焦纯四足 / UAV / 移动操作；**REBot**（arXiv:2508.06229）面向足式反射规避，**未覆盖轮足混合动力学与高速滚动**。
- **主张 AWARE：** 分层架构 — 高层威胁感知策略输出速度指令并选择模态；低层两个预训练专家分别服务导航级避让与高动态反射；两阶段 PPO + 课程学习。
- **贡献三点：** (i) 形式化轮足高动态反射规避问题并提出 AWARE；(ii) 缓解模态混淆、涌现多样步态与导航↔反射切换；(iii) Isaac Lab + M20 真机多样动态场景验证。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-aware-wheeled-legged-reflexive-evasion.md`](../../wiki/entities/paper-aware-wheeled-legged-reflexive-evasion.md)；交叉 [轮足四足](../../wiki/concepts/wheel-legged-quadruped.md)、[Hybrid Locomotion](../../wiki/tasks/hybrid-locomotion.md)、[分层 RL](../../wiki/methods/hierarchical-reinforcement-learning.md)、[MUJICA](../../wiki/entities/paper-mujica-wheel-legged-multi-skill.md)。

## 摘录 2：分层架构与双专家（§IV-A）

- **激活条件：** 相对速度 / 相对距离超阈值时触发规避策略：\(\alpha_t=\mathbb{I}(\|v_{\mathrm{rel}}\|>v_{\mathrm{th}}\lor\|p_{\mathrm{rel}}\|<d_{\mathrm{th}})\)。
- **相对逼近率 RAR：** \(\kappa=-d|p_{\mathrm{rel}}|/dt=-(p_{\mathrm{rel}}^{\top}v_{\mathrm{rel}})/|p_{\mathrm{rel}}|\)，作威胁量化特征。
- **高层：** 观测 = \(\kappa\)、\(p_{\mathrm{rel}}\)、本体状态 \(x^t\)；输出 \(v_{\mathrm{cmd}}\in\mathbb{R}^3\)；末隐层分支 **Gumbel-Softmax** 得 one-hot \(m=[m_{\mathrm{low}},m_{\mathrm{high}}]^{\top}\)。
- **低层：** 观测 = \(v_{\mathrm{cmd}}^{t-4:t}\)、本体 \(x^t\)；动作 \(a^t=m_{\mathrm{low}}\pi_{\mathrm{low}}+m_{\mathrm{high}}\pi_{\mathrm{high}}\)（确定性路由，非 MoE 软混合）。
  - \(\pi_{\mathrm{low}}\)：低速全向跟踪，服务导航级避让。
  - \(\pi_{\mathrm{high}}\)：高加速前冲/侧向机动；历史指令用于急减速时调 pitch 与轮阻尼制动。
- **与 MoE / teacher-student：** 明确用 **两专用专家 + 硬切换**，避免模式混淆。

**对 wiki 的映射：** 实体页画流程总览；HRL 页补「速度指令 + 离散专家路由」实例。

## 摘录 3：两阶段训练、奖励与 Sim2Real（§IV-B / IV-C）

- **S1：** 独立训 \(\pi_{\mathrm{low}}\)（\(\|a\|\leq 1.5\,\mathrm{m/s}^2\)）与 \(\pi_{\mathrm{high}}\)（\(a_x\in[1.0,5.5]\)，\(a_y\in[1.0,3.0]\,\mathrm{m/s}^2\)）；Isaac Lab + PPO。
- **S2：** 冻结低层；高层按规避成功率课程升高障碍飞行速度；奖励含任务成功/碰撞/假阳性惩罚、跟踪可行性、平滑、能量。
- **威胁指示 \(\xi\)：** \(\kappa>\kappa_{\mathrm{th}}\) 且预测最小距离 \(<d_{\mathrm{safe}}\) 才算真威胁；近静/非中心障碍触发规避则罚 \(\|v_{\mathrm{cmd}}\|^2\)。
- **域随机化（TABLE II）：** 障碍距离、静/非中心比例、初始位姿/角速度、执行器增益、外扰、COM、惯量、质量、摩擦等。
- **真机：** 动捕测机器人与障碍；场景 = 抛箱、带盒长杆戳、人脚踢；多方向冲击。

**对 wiki 的映射：** [Sim2Real](../../wiki/concepts/sim2real.md) / [Domain Randomization](../../wiki/concepts/domain-randomization.md) / [Curriculum Learning](../../wiki/concepts/curriculum-learning.md) 补轮足高动态避障案例指针。

## 摘录 4：评测要点（§V–VI）

| 设定 | 要点 |
|------|------|
| 仿真指标 | ASR（避障成功率）、AMD（规避位移）；反应时间 \(t_r\in[0,3.5]\,\mathrm{s}\)；全向弹体来袭 |
| 基线 | **REBot**（足式反射规避）、**RB-MPC**（Relaxed Barrier MPC）；AWARE 在 ASR/AMD 上更优，尤其短反应时间 |
| TABLE III | \(t_r\) 0–1 s：ASR \(0.294\pm0.315\)；1–2 s：\(0.931\pm0.058\)；2–3 s：\(0.973\pm0.027\) |
| 消融 | 去课程 / 去双专家 / 去任务奖励 / 去可行性正则均伤 ASR；**无双专家** 模态混淆与极限威胁下失衡最严重 |
| 极限启动 | Rolling vs Stepping：TEA 4.13 vs 2.78 m/s²；TNA 3.678 vs 1.251；滚动更偏轮功率，踏步更偏 hip-X/knee |
| 行为模式 | 运动学：Stepping / Rolling / Hybrid；策略层：Navigation Avoidance / Reflexive Evasion；高动态涌现 Forward Lunge、Lateral Dodge |
| 真机 | 反射规避平均 ASR **59%**、AMD **1.1 m**；受 M20 硬件上限与 sim2real gap 限制；混合连续场景可导航→反射切换 |

**对 wiki 的映射：** 结论节强调「双专家硬切换 + RAR/课程」相对足式 REBot 移植到轮足的必要性；真机 ASR 读硬件与感知边界，勿当仿真同量级。

## 对 wiki 的映射（汇总）

- 新建实体页：[AWARE（arXiv:2604.23761）](../../wiki/entities/paper-aware-wheeled-legged-reflexive-evasion.md)
- 交叉补强：[轮足四足](../../wiki/concepts/wheel-legged-quadruped.md)、[Hybrid Locomotion](../../wiki/tasks/hybrid-locomotion.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[分层 RL](../../wiki/methods/hierarchical-reinforcement-learning.md)、[MUJICA](../../wiki/entities/paper-mujica-wheel-legged-multi-skill.md)、[robot_lab](../../wiki/entities/robot-lab.md)（M20 仿真入口）
