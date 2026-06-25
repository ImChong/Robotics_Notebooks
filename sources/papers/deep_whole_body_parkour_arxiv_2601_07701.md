# Deep Whole-body Parkour（arXiv:2601.07701）

> 来源归档（ingest）

- **标题：** Deep Whole-body Parkour
- **类型：** paper / humanoid / perceptive locomotion / whole-body tracking / parkour / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2601.07701>
- **arXiv HTML：** <https://arxiv.org/html/2601.07701>
- **PDF：** <https://arxiv.org/pdf/2601.07701>
- **项目页：** <https://project-instinct.github.io/deep-whole-body-parkour>
- **机构：** 清华大学交叉信息研究院（IIIS, Tsinghua）、上海期智研究院（Shanghai Qi Zhi Institute / Shanghai PIL）
- **硬件：** Unitree G1；头部深度感知
- **入库日期：** 2026-06-25
- **一句话说明：** 把 **外感知深度** 接入 **全身运动跟踪（WBT）**：单策略在多类跑酷动作与多样障碍几何上闭环调整落脚/撑手时机，实现 vault、dive-roll 等非纯足式穿越；自定义 Warp **分组射线投射** 支撑万级并行深度仿真。

## 摘要级要点

- **问题：** 感知 locomotion 限于 **足式踏板**；盲 WBT/AMP 跟踪 **环境无关**，无法按障碍高度/距离调整 vault 时机与手部位。
- **数据：** 动捕专家在真实障碍上表演 + iPad LiDAR 同步场景网格；**GMR** retarget 至 G1；剔除实验室墙壁，保留 **运动–地形对** 在 Isaac Lab 开放场随机摆放。
- **相对参考系跟踪：** $T_{\text{rel}}$ 取机器人 xy/yaw + 参考 z/roll/pitch，解耦平面跟踪与垂直/俯仰模仿（BeyondMimic/VMP 类奖励：根位姿、关键连杆相对/全局速度等）。
- **感知接口：** actor：**1 帧噪声深度 + 8 帧本体历史** + 同步未来参考关节/根位姿；critic 特权含 **地形 height-scan**、连杆真值等。
- **自适应采样：** 基于失败率的课程，难动作/地形对过采样（Algorithm 2）。
- **仿真工程：** 预计算 collision group → mesh 索引表；射线仅查 **全局静态地形 + 本 agent 动态连杆**，相对朴素实现 **~10×** 深度渲染加速。
- **能力：** 非结构化地形上 **撑跳、俯冲翻滚** 等多接触高动态技能；对初始位姿偏差 **闭环纠偏**（相对盲跟踪需精确摆放）。

## 核心摘录（面向 wiki 编译）

### 与 Project Instinct / 跑酷簇对照

| 维度 | Deep WBP（本文） | PHP / Hiking（Instinct 姊妹） | 盲 WBT |
|------|------------------|-------------------------------|--------|
| 感知 | **深度闭环** | 深度跑酷/徒步 | 无 |
| 技能 | **全身多接触 vault 等** | 踏板跑酷为主 | 固定轨迹 |
| 训练对 | **动捕–网格配对** | 类似 Instinct 管线 | AMASS 平地 |
| AMP | **非典型 AMP 主线** | 专题 #09 Hiking 等 | AMP 风格先验 |

### 策展导读（AMP 专题 #18 / RL 栈 #23）

- 非 AMP 核心论文，但属 **交互与长时程** 段：**感知 × 全身跟踪** 扩展 traversability。
- 与 [Embrace Collisions #19](../../wiki/entities/paper-amp-survey-19-embrace_collisions.md) 同属 [Project Instinct](../../wiki/entities/project-instinct.md)。

## 对 wiki 的映射

- 沉淀实体页：[Deep Whole-body Parkour](../../wiki/entities/paper-deep-whole-body-parkour.md)
- 交叉：[project-instinct](../../wiki/entities/project-instinct.md)、[stair-obstacle-perceptive-locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[humanoid-rl-motion-control-body-system-stack](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[paper-hiking-in-the-wild](../../wiki/entities/paper-hiking-in-the-wild.md)

## 参考来源（原始）

- arXiv:2601.07701
- [humanoid_amp_survey_18_deep_whole_body_parkour.md](humanoid_amp_survey_18_deep_whole_body_parkour.md)
- [humanoid_rl_stack_23_deep_whole_body_parkour.md](humanoid_rl_stack_23_deep_whole_body_parkour.md)
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
