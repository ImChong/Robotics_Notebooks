# HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control（arXiv:2602.03205）

> 来源归档（ingest）

- **标题：** HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control
- **类型：** paper / humanoid / skateboarding / AMP / hybrid contact / sim2real
- **venue：** Robotics: Science and Systems（RSS）2026
- **arXiv abs：** <https://arxiv.org/abs/2602.03205>
- **arXiv HTML：** <https://arxiv.org/html/2602.03205>
- **PDF：** <https://arxiv.org/pdf/2602.03205>
- **项目页：** <https://husky-humanoid.github.io/>
- **代码：** <https://github.com/TeleHuman/humanoid_skateboarding>
- **机构：** 中国电信人工智能研究院（TeleAI）、香港大学（HKU）、中国科学技术大学（USTC）、上海科技大学（ShanghaiTech）、上海交通大学（SJTU，作者隶属以论文为准）
- **硬件：** Unitree G1（23 可控 DoF，不含腕部）
- **入库日期：** 2026-06-25
- **一句话说明：** 显式建模滑板 **倾角–转向角耦合** $\tan\sigma=\tan\lambda\sin\gamma$，混合 **推地（AMP 人形蹬地）+ 倾身转向 + 轨迹引导相位切换**，G1 真机完成室内外滑板推进与转向。

## 摘要级要点

- **问题：** 人形滑板 = 高 CoM、小支撑面、侧向转向、腿重定向 + **欠驱动轮板** 强耦合；四足滑板经验难以直接迁移（Table I 对比 G1 vs Go1）。
- **系统建模：** 简化 kingpin 几何，导出板倾 $\gamma$ 与 truck 转向 $\sigma$ 等式约束；任务表述为 **推地 / 滑行转向** 两相周期性混合动力系统。
- **观测：** actor 5 步本体历史 $\bm{o}^{\text{prop}}_{t-4:t}$（78 维）+ 命令 $[v_{cmd},\psi]$ + 相位 $\Phi=(t\bmod H)/H$；critic 特权含滑板位姿、倾角、足–地/足–板接触力等。
- **推地相 AMP：** 判别器输入 5 步关节角窗 $\tau_t$；风格奖励 $r^{\text{style}}=\alpha\max(0,1-\frac{1}{4}(d-1)^2)$；$r^{\text{push}}=r^{\text{task}}+r^{\text{style}}$。
- **转向相：** 运动学引导目标倾角，利用式 (1) 耦合实现 **lean-to-steer**；自行车模型近似偏航率 $\dot\psi=\frac{v}{L}\tan\sigma$。
- **相位切换：** 轨迹规划机制平滑推地↔转向；总奖励按相位指示器 $\mathbb{I}^{\text{push/steer/trans}}$ 组合。
- **真机：** 室内外多滑板泛化、抗扰动；RSS 2026 正式发表。

## 核心摘录（面向 wiki 编译）

### 与相邻人形–物体交互对照

| 维度 | HUSKY（本文） | PhysHSI（#15） | HUSKY 四足先例 [19] |
|------|-------------|----------------|---------------------|
| 平台动力学 | **欠驱动轮板 + 非完整约束** | 静态场景物体 | 四足低 CoM |
| AMP 作用相 | **仅推地蹬地风格** | 全程 HSI 风格 | 推地风格 |
| 控制接口 | 速度 + 航向 + 周期相位 | 物体/目标位姿 | 速度命令 |
| 物理先验 | **倾角–转向显式耦合** | 接触帧标注物体 | 简化滑板模型 |

### 策展导读（AMP 专题 #14）

- AMP 在此 **不服务全身所有相位**，而是给 **推地人形蹬地** 注入风格，转向靠 **物理约束奖励** — 混合先验 + 物理引导的范例。
- 与 #13 动态球交互、#15 场景 HSI 并列 **04 交互与长时程**。

## 对 wiki 的映射

- 沉淀实体页：[HUSKY（AMP #14）](../../wiki/entities/paper-amp-survey-14-husky.md)
- 交叉：[amp-reward](../../wiki/methods/amp-reward.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[PhysHSI #15](../../wiki/entities/paper-amp-survey-15-physhsi.md)、[MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md)（同 TeleAI/Bai 系感知 locomotion）

## 参考来源（原始）

- arXiv:2602.03205 — 论文正文
- [humanoid_amp_survey_14_husky_humanoid_skateboarding_system_via_physics.md](humanoid_amp_survey_14_husky_humanoid_skateboarding_system_via_physics.md)
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
