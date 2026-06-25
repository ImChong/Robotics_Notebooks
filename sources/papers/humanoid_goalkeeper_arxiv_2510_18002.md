# Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints（arXiv:2510.18002）

> 来源归档（ingest）

- **标题：** Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints
- **类型：** paper / humanoid / dynamic object interaction / AMP / goalkeeper / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2510.18002>
- **arXiv HTML：** <https://arxiv.org/html/2510.18002v2>
- **PDF：** <https://arxiv.org/pdf/2510.18002>
- **项目页：** <https://humanoid-goalkeeper.github.io/Goalkeeper/>
- **代码：** <https://github.com/InternRobotics/Humanoid-Goalkeeper>
- **机构：** 香港大学（HKU）、上海人工智能实验室（Shanghai AI Lab）；一作 Ren 实习于上海 AI Lab 具身智能中心
- **硬件：** Unitree G1（29 DoF 策略输出）；Intel RealSense D435i 机载深度相机 **或** 光学动捕
- **入库日期：** 2026-06-25
- **一句话说明：** 单阶段端到端 PPO：以球落点区域 $\mathcal{R}$ 条件化 **任务奖励 + 多判别器 position-conditioned AMP**，在宽守门区内实现人形全身自然扑救；支持 MoCap / 机载相机双感知模态，并泛化至躲球、抓球。

## 摘要级要点

- **问题：** 四足守门已有先例，但人形守门需 **更大横向覆盖 + 更短反应时间 + 全身高动态**；固定轨迹跟踪或遥操作难以兼顾任务与自然性。
- **单策略端到端：** actor 输入含 **局部坐标系球位置** $\mathbf{O}_{\text{ball}}$、本体历史（$T$ 步）与 29 维关节状态；critic 特权项含球速、左右手位置、落点区域 $\mathcal{R}$、末端目标 $\mathbf{p}_{\text{target}}$ 等。
- **Position-conditioned task rewards：** 球门线划分为 $k$ 个落点区域；动态末端目标在 **预测落点** 与 **当前球位** 间切换（距离阈值 $d_{th}$）；区域调制项 $\nu^{(\mathcal{R})}$ 鼓励侧移/起跳等全身动作；episode 3 s 含 **扑救后稳定** 奖励与 **非默认姿态重置**（连续扑救）。
- **Position-conditioned AMP：** 每区域 $\mathcal{R}$ 独立参考动作槽 + 判别器 $D^{(\mathcal{R})}$；参考来自自录 RGB 视频经 **GVHMR** 提取并 retarget 至 G1；**软约束 AMP**：对执行转移 $(q_t,q_{t+1})$ 高斯扰动后取 **最高判别分** 样本计奖，减轻与高精度任务冲突。
- **感知 sim2real：** 训练内嵌 **球位置/区域估计器**（MSE + CE）；球位噪声 ±5 cm、0.4 s 后随机 dropout、球停后观测置零；真机支持 MoCap 标记或 D435i+IR 滤光高反光球检测。
- **评测：** 仿真 6 区域 × 500 trials；默认飞行 0.5–1.0 s、落距 3–5 m、门宽 ±1.5 m；**$E_{\text{succ}}\approx 80.9\%$**（完整方法）；真机 5 人制门（3 m×2 m）MoCap **21/30** 成功。

## 核心摘录（面向 wiki 编译）

### 与相邻 AMP / 动态交互路线对照

| 维度 | Humanoid Goalkeeper（本文） | 四足守门 [12] | MoRE（#08） | PhysHSI（#15） |
|------|---------------------------|---------------|-------------|----------------|
| 先验路由 | **落点区域 $\mathcal{R}$** | 区域技能选择 | **gait command** | 单判别器 + 物体位姿进判别观测 |
| 任务 | **毫秒级飞球拦截** | 飞球拦截 | 复杂地形步态 | 搬箱/坐躺站 |
| 跟踪形态 | **软 AMP，非相位跟踪** | AMP 分区域 | 多判别器硬门控 | RSI + 长时程 HSI |
| 感知 | 球位局部观测 + 噪声 | 同类 | 深度地形 | LiDAR+相机物体定位 |

### 仿真消融（Table II 摘要）

- 去任务约束：$E_{\text{succ}}\downarrow 9.4\%$，动作最平滑但无效。
- 去运动约束：$E_{\text{succ}}\approx 31.7\%$，$E_{\text{match}(\mathcal{R})}$ 崩塌。
- 去 AMP 分区：$E_{\text{match}(\mathcal{R})}\approx 25.7\%$（完整 67.8%）。
- 完整方法在 Range-Easy（±1.0 m）达 **84.6%** 成功率。

### 策展导读（AMP 专题 #13）

- 交互段代表：**任务条件化多判别器 AMP** — 把「像人」绑定到 **扑救语义区域**，而非单一守门风格。
- 与 #12 HAML、#15 PhysHSI 同属 **04 交互与长时程**，但强调 **动态物体 + 极短反应窗**。

## 对 wiki 的映射

- 沉淀实体页：[Humanoid Goalkeeper（AMP #13）](../../wiki/entities/paper-amp-survey-13-humanoid_goalkeeper.md)
- 交叉：[amp-reward](../../wiki/methods/amp-reward.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)、[humanoid-amp-motion-prior-survey](../../wiki/overview/humanoid-amp-motion-prior-survey.md)、[PhysHSI #15](../../wiki/entities/paper-amp-survey-15-physhsi.md)、[MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md)

## 参考来源（原始）

- arXiv:2510.18002 — 论文正文
- [humanoid_amp_survey_13_humanoid_goalkeeper_learning_from_position_condi.md](humanoid_amp_survey_13_humanoid_goalkeeper_learning_from_position_condi.md) — AMP 19 篇策展索引
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
