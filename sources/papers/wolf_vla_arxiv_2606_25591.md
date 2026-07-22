# WOLF-VLA: Whole-Body Humanoid Optimal Locomotion Framework for Vision-Language-Action Learning

> 来源归档（ingest）

- **标题：** WOLF-VLA: Whole-Body Humanoid Optimal Locomotion Framework for Vision-Language-Action Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.25591>
- **机构：** DFKI Robotics Innovation Center；DFKI Interactive Machine Learning；University of Oldenburg；University of Bremen
- **入库日期：** 2026-07-03
- **更新日期：** 2026-07-22
- **开源状态：** 待发布；摘要承诺 full dataset、model checkpoints、benchmarking simulation suite 将开放，但截至 2026-07-22 未发现官方仓库或下载页。
- **一句话说明：** 用 **全身最优控制（OCP）** 合成六类 locomotion 相关任务的动态一致轨迹数据集（277h / 15,276 episodes），同步 ego 视觉与自然语言训练 **VLA**；强调数据最优性、安全性、接触一致性和可复现 benchmark。

## 摘要要点

- 问题：现有 VLA 多聚焦固定基座或机械臂，对 contact-rich humanoid whole-body locomotion 缺少动态一致数据和 benchmark。
- 核心：用 OCP 生成满足多体动力学、接触、关节/速度/扭矩限制的全身轨迹。
- 数据：生成 277h humanoid locomotion，覆盖 forward walking、side gait、stairs、turning、squatting 等任务。
- 模态：每条样本包含 OC joint trajectory、ego RGB observation 和 natural-language instruction。
- 策略：从 GR00T-N1.5-3B 初始化 fine-tune action diffusion model，建立 WOLF-VLA baseline。

## 方法要点

- **OCP formulation：** 多相 contact schedule；成本包括 CoM、feet、torque minimization、posture regularization；约束包括 joint/velocity/torque limits。
- **Solver：** Crocoddyl + Pinocchio + Box-FDDP；每条 motion 还经 OCP success check 与 manual validation。
- **Environment variation：** target shapes、six colors、约 40×40 spatial placements、随机 distractors。
- **Virtual environment：** MuJoCo/Gymnasium；RH5 humanoid，free-flyer + 25 actuated joints；head camera 120° FOV，224×224 RGB。
- **Language generation：** 从 OCP metadata 自动生成自然语言，附 spatial/height structured tags。
- **VLA training：** GR00T-N1.5-3B 初始化，freeze LLM/vision encoder，训练 action diffusion 与 projector layers；LeRobot format。

## 实验与数字

- **数据总量：** 277 h，15,276 episodes，平均 episode length 28 s。
- **任务 episode：** WF 2874；WA 8234；W.CS.U 2358；W.CS.U/D 1810。
- **训练资源：** 4×A100，200,000 gradient steps，effective batch size 128，bfloat16。
- **主模型成功率：** WF 99%，WA 27%，W.CS.U 51%，W.CS.U/D 44%，All 55.3%。
- **baseline：** ACT all average 1.4%，π0.5 0%。
- **模态消融：** removing vision 导致最严重下降；spatial tags/language 对复杂任务有帮助但不是唯一 grounding。

## 开源 / 复现状态

- **代码：** 未发现官方仓库。
- **数据/checkpoints：** 摘要承诺将公开 full dataset、model checkpoints 和 simulation suite，但尚未发布。
- **项目页：** 未发现官方项目页。
- **复现边界：** 需要 RH5 模型、OCP task definitions、Crocoddyl/Pinocchio 配置、MuJoCo environment、LeRobot conversion、GR00T fine-tuning 配置。

## 对 wiki 的映射

- [paper-wolf-vla](../../wiki/entities/paper-wolf-vla.md) — 完整实体页，含 OC 数据管线、VLA 训练、评测和开源状态。

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2606.25591>
- 接触横切面编译：[wechat_embodied_ai_lab_loco_manip_contact_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
