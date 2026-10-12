# ATHLETE（athlete-humanoid-tennis，匿名镜像）

> 来源归档（repo；核对日期：2026-10-12；通过 anonymous.4open.science 文件 API 读取 README/LICENSE，未整库克隆）

- **代码：** <https://anonymous.4open.science/r/athlete-humanoid-tennis>
- **论文：** <https://arxiv.org/abs/2610.13572>；**项目页：** <https://hazintihoa.github.io/athlete.github.io/>
- **许可：** MIT（署名 Anonymous Authors，部分代码派生自 TaskNPoint，保留其许可）
- **沉淀到 wiki：** [paper-athlete-reactive-humanoid-tennis-tgmm.md](../../wiki/entities/paper-athlete-reactive-humanoid-tennis-tgmm.md)

## 发布状态（README 自述，分阶段发布）

- **已发布：** 训练代码（时间自适应教师、TPPO 学生蒸馏 + PPO 微调、全程飞行微调，`athlete/`）；仿真与评测（native/viser 查看器、ONNX 回放）；MuJoCo 与真机 Unitree G1 的 ROS 2 Jazzy 部署栈（`deploy/`）；已发布学生策略 `deploy/policies/m14_11/`（`model_36499.pt` + `.onnx`）；冻结教师 `checkpoints/teacher_model_29999.pt`；G1+球拍模型（`robots/`）；164 段视频击球片段（`data/djvkovic_npz_data_v5/`）；TGMM 数据库构建与参考生成（`motion_matching/`）；LAFAN1 转换脚本（第三方动作数据不再分发）。
- **未发布（Coming soon）：** 训练用参考动作数据集 `rollout_1000epis` / `rollout_1000epis_relabelled`（各约 430 MB）与全程飞行微调检查点。TGMM 第三步「模拟模仿动态修正」不含在仓库内。

## 入口

- 训练（Python 3.13 + uv + CUDA 12.8）：`uv run train Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs ...`（教师，8192 环境，30000 迭代）；`uv run train Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-...-Unitree-G1 --agent.algorithm.teacher-checkpoint checkpoints/teacher_model_29999.pt`（学生，TPPO 在 `athlete/src/athlete/goal_cond_tracking/rl/tppo.py`）。
- 回放：`uv run python -m athlete.scripts.play <task> --checkpoint-file deploy/policies/m14_11/model_36499.pt --viewer viser`。
- TGMM：`motion_matching/build_official_mm_db.py` 建库，`motion_matching/tennis_official_mm_mj.py --collect 1000 --headless` 生成参考。
- 部署：`bash run_tppo.sh sim g1_tppo_student_m14_11.yaml`；真机需 `estimation/mocap/natnet_policy_bridge.py`（OptiTrack NatNet `G1_pelvis`、`Ball` 刚体）并设 `TPPO_ENABLE_REAL_ROBOT=YES`。
- 策略接口：ONNX 输入 `obs` 631 维（当前观测 136 + 状态历史 465 + 球历史 30），输出 `actions` 29 维；控制周期 20 ms，物理步 2.5 ms、decimation 8；训练中球观测延迟 20±10 ms。
- README 明确该策略包未经无人值守真机运行验证。本次未实际运行训练或部署。
