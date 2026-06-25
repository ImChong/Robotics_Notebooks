# Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids（arXiv:2601.07718）

> 来源归档（ingest）

- **标题：** Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids
- **类型：** paper / humanoid perceptive locomotion / depth E2E RL / sim2real / open-source
- **arXiv abs：** <https://arxiv.org/abs/2601.07718>
- **arXiv HTML：** <https://arxiv.org/html/2601.07718v1>
- **PDF：** <https://arxiv.org/pdf/2601.07718>
- **项目页：** <https://project-instinct.github.io/hiking-in-the-wild>
- **机构：** 清华大学交叉信息研究院（IIIS）；上海期智研究院；清华大学计算机系
- **入库日期：** 2026-06-25
- **一句话说明：** **单阶段 E2E RL**：原始 **深度图 + 本体** 直接映射关节目标；**地形边缘检测 + 足端体积点** 软约束防踩边滑倒，**Flat Patch Sampling** 生成可达导航目标防 reward hacking；野外真机最高 **2.5 m/s**，训练部署代码开源。

## 摘要级要点

- **问题：** 盲 locomotion 反应式、难预见深沟高台；建图/LiDAR 依赖定位漂移；既有深度方案多低速、难扩展、少开源。
- **观测：** 本体历史 $h$ 步 + 深度历史 $\mathcal{H}_t$；非对称 critic 含真值线速度等特权信息。
- **动作：** 29 维关节目标 + PD（增益沿用 BeyondMimic）。
- **深度仿真：** NVIDIA **Warp** 并行 ray-cast；$\mathcal{F}_{sim}$ / $\mathcal{F}_{real}$ 双向对齐噪声分布，零样本 sim2real。
- **安全机制：** **Terrain Edge Detection**（任意 trimesh 自动提边）+ **Foot Volume Points** 穿透边缘惩罚 → 隐式学「脚踩实面中心」。
- **命令：** Flat Patch Sampling 在网格上采可达平面块，相对位置生成速度命令（含随机限速），避免原地转圈 hack。
- **架构：** 策略网络含 **MoE**；总奖励 $R=r_{\mathrm{task}}+r_{\mathrm{reg}}+r_{\mathrm{safe}}+r_{\mathrm{amp}}$（含 AMP-style 自然性项）。
- **真机：** 机载前向深度 **60 Hz**；无外部定位；野外楼梯、坡地、草地、离散障碍。
- **消融（仿真 10 类地形）：** 去 MoE / 去深度历史 / 去 pose-based 命令 / 去 AMP 均降成功率；Small Box 无 AMP **0%** vs 完整 **99.09%**。

## 核心摘录（面向 wiki 编译）

### 在双索引中的位置

| 索引 | 编号 | 层/段 |
|------|------|-------|
| [42 篇 RL 身体系统栈](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) | 24/42 | 03 感知式高动态运动 |
| [AMP 19 篇专题](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md) | 09/19 | 02 人形走跑（非典型 AMP，但同属走跑/感知交叉） |

## 对 wiki 的映射

- 沉淀实体页：[paper-hiking-in-the-wild.md](../../wiki/entities/paper-hiking-in-the-wild.md)
- 任务语境：[stair-obstacle-perceptive-locomotion.md](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)
- 姊妹篇：[MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md)、[PHP / Deep Parkour 等](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 策展索引：[humanoid_amp_survey_09_…](humanoid_amp_survey_09_hiking_in_the_wild_a_scalable_perceptive_parkour.md)、[humanoid_rl_stack_24_…](humanoid_rl_stack_24_hiking_in_the_wild_a_scalable_perceptive_parkour.md)

## 参考来源（原始）

- arXiv:2601.07718 — 论文正文
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)、[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
