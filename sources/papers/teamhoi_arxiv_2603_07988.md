# TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size（arXiv:2603.07988）

> 来源归档（ingest）

- **标题：** TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size
- **类型：** paper / humanoid / multi-agent / HOI / masked-AMP / transformer
- **venue：** IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR）2026
- **arXiv abs：** <https://arxiv.org/abs/2603.07988>
- **arXiv HTML：** <https://arxiv.org/html/2603.07988>
- **PDF：** <https://arxiv.org/pdf/2603.07988>
- **项目页：** <https://splionar.github.io/TeamHOI/>
- **代码：** <https://github.com/sail-sg/TeamHOI>
- **机构：** Garena、东南亚人工智能实验室（Sea AI Lab）、新加坡国立大学（NUS）
- **入库日期：** 2026-06-25
- **一句话说明：** **单一去中心化 Transformer 策略** + **队友 token 交叉注意力**，配合 **masked AMP**（交互部位用 $D_{\text{mask}}$、非交互用 $D_{\text{full}}$），在 2–8 人协作搬桌任务上零重训泛化队形与桌面几何。

## 摘要级要点

- **问题：** 协作 HOI 缺多人体 MoCap；固定 MLP 输入维度无法 **可变队形**；CooHOI 等依赖共享物体动力学作隐式通信、且每队形独立策略。
- **TeamHOI 框架：** 各 agent 局部观测 $o_t=(s_t,g_t)$ token 化；**self-attention + cross-attention** 读队友 tokens $\{\mathcal{T}_t^i\}$；并行实例化不同人数环境，**按队形分别归一化 PPO advantage**。
- **Masked AMP：** 全身判别器 $D_{\text{full}}$ + **遮蔽手/前臂等交互部位**的 $D_{\text{mask}}$；风格奖励 $r^{\text{style}}=\sigma(\alpha_t)r^{\text{mask}}+(1-\sigma(\alpha_t))r^{\text{full}}$，$\alpha_t$ 为连续交互指示（如人–物距离）。
- **协作搬桌任务：** 随机初始位姿 → 自主围桌 **formation**（64 周边接触候选，无 oracle 手位分配）→ 抬起运输 → 放下；桌面方形/矩形/圆形。
- **Formation reward：** 角向均匀 $r_{\text{ang}}$ + 主轴覆盖 $r_{\text{cov}}$（凸包支撑多边形相对桌 CoM 主方向覆盖）；$r_{\text{form}}=0.25 r_{\text{ang}}+0.75 r_{\text{cov}}$，对队形与形状 **agnostic**。
- **评测：** 单策略在 2/4/8 人及 OOD 队形高成功率；相对 CooHOI-*2/4/8 三策略显著更优（项目页 rollout 对比）。

## 核心摘录（面向 wiki 编译）

### 与相邻多智能体 / AMP 对照

| 维度 | TeamHOI（本文） | CooHOI | PhysHSI（#15） |
|------|----------------|--------|----------------|
| 队形 | **单策略任意 N** | 每 N 一策略 / 无队友状态 | 单人 |
| AMP 数据 | **单人参考 + mask** | 单人全身 | 单人–物体 MoCap |
| 协调 | **队友 token Transformer** | 物体间接耦合 | N/A |
| 任务 | 协作搬桌 | 协作搬物 | 搬箱/坐躺站 |

### 策展导读（AMP 专题 #17）

- **masked AMP** 解决「没有协作 MoCap 但仍要自然非交互肢体」— 与 MoRE **多判别器分 gait**、Goalkeeper **分区域判别器** 同属 **条件化/分部位先验** 家族。
- CVPR 2026；Sea AI Lab / TokenHSI 脉络（Transformer 统一 HSI）。

## 对 wiki 的映射

- 沉淀实体页：[TeamHOI（AMP #17）](../../wiki/entities/paper-amp-survey-17-teamhoi.md)
- 交叉：[amp-reward](../../wiki/methods/amp-reward.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[PhysHSI #15](../../wiki/entities/paper-amp-survey-15-physhsi.md)、[humanoid-amp-motion-prior-survey](../../wiki/overview/humanoid-amp-motion-prior-survey.md)

## 参考来源（原始）

- arXiv:2603.07988
- [humanoid_amp_survey_17_teamhoi_learning_a_unified_policy_for_cooperativ.md](humanoid_amp_survey_17_teamhoi_learning_a_unified_policy_for_cooperativ.md)
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
