# php_parkour（PHP 官方实现）

- **类型：** repo / motion matching / perceptive humanoid parkour
- **代码：** <https://github.com/amazon-far/php_parkour>
- **项目页：** <https://php-parkour.github.io/>
- **浏览器演示：** <https://php-parkour.github.io/demo.html>
- **论文：** <https://arxiv.org/abs/2602.15827>（RSS 2026）
- **许可证：** Apache-2.0（仓库代码；使用外部依赖或动作资产时仍需核查其许可）
- **入库 / 核查日期：** 2026-10-10
- **源码快照：** `554edcc33393fbd8bd2a725a781ffb867f6c467a`
- **依赖快照：** `thirdparty/holosoma` gitlink `4a0bf04cde5930af5ed59da797568d32be972525`，不要直接用 Holosoma 最新 main 替代。
- **一句话说明：** 官方已公开离线 motion matching、IsaacSim 专家训练、WARP 深度学生蒸馏与 MuJoCo Sim2Sim；Release 同时提供动作数据库、五组预制训练数据及学生 ONNX 双模型。

## 官方入口与原始资料

以下路径均核对上述源码快照；仓库 `main` 链接用于继续跟进，复现时应记录 commit。

| 文件 / 目录 | 官方文档或实现 | 用途 |
|---|---|---|
| `README.md` | [总入口](https://github.com/amazon-far/php_parkour/blob/main/README.md) | 递归克隆、三阶段流程、学生下载 |
| `motion_matching/README.md` | [动作生成](https://github.com/amazon-far/php_parkour/blob/main/motion_matching/README.md) | 数据库下载、场景生成、可视化与自定义技能 |
| `wbt_training/README.md` | [训练指南](https://github.com/amazon-far/php_parkour/blob/main/wbt_training/README.md) | IsaacSim 5.1、数据 registry、教师与学生训练、评估 / 导出 |
| `wbt_training/DEPLOY.md` | [Sim2Sim 指南](https://github.com/amazon-far/php_parkour/blob/main/wbt_training/DEPLOY.md) | 两进程、共享内存、图形桌面与操作按键 |
| `run_php_sim.sh` / `run_php_inference.sh` | [仿真入口](https://github.com/amazon-far/php_parkour/blob/main/run_php_sim.sh) / [推理入口](https://github.com/amazon-far/php_parkour/blob/main/run_php_inference.sh) | 锁定 Holosoma 的 G1 / D435i 预设 |
| `scripts/download_assets.py` / `release-assets.json` | [下载器](https://github.com/amazon-far/php_parkour/blob/main/scripts/download_assets.py) / [资产清单](https://github.com/amazon-far/php_parkour/blob/main/release-assets.json) | 按组件下载，核对大小及 SHA-256；`--verify` 验证本地缓存 |

## 开放边界（核查结果）

- **代码已公开：** motion matching、教师训练、学生 DAgger / DAgger+PPO、ONNX 导出及本地深度推理入口；不是只有浏览器 demo。
- **数据库已公开：** `databases` 组件提供 locomotion、climb、vault、roll 等 motion matching 数据库与地形元数据；发布标签 `motion-matching-assets-v1`。
- **预制训练示例已公开：** `training-motion-examples-v1` 包含 locomotion（100 对）、low-step（56 对）、high-step（60 对）、low-climb-76（40 对）、high-climb-76（40 对），共 **296 对 motion NPZ / terrain NPY**。每个数据集单独训练教师，再按对应顺序提供给学生；不能据此宣称论文所有技能的数据与教师均已齐备。
- **推理权重已公开：** `student` 组件是配套的 `depth_backbone.onnx` 与 `student.onnx`，README 标注为 sanitized 导出；不是原始学生 `.pt` 或已训练教师 checkpoint。
- **真机与实验边界：** 官方部署文档当前给出的是本地 MuJoCo + 深度共享内存路线，不是无需标定 / 安全调试的真机一键复现包。论文的 1.25 m 攀墙等指标不能视为这五组训练示例的实测结果。

## 从 README 到代码的流程摘录

1. **离线合成：** `motion_matching.run --mode generate --scenario high_speed_climb_76`，使用数据库合成 locomotion–skill–locomotion；输出 50 FPS 的 `*_motion.npz`、`*_terrain.npy` 与 `*_terrain.obj`。源动作输入的帧率与内部数据库重采样帧率不同，不应统一写成 50 FPS。
2. **教师：** `wbt_training/training_runs/run_terrain_teacher.sh` → `wbt_training.train_agent` → Holosoma `TrainingContext` / IsaacSim；`REGISTRY` 指向本地 `file://` 数据集或 W&B registry。
3. **学生：** `run_terrain_warp_distill.sh` 默认 `FINETUNE=1`（DAgger+PPO）；`FINETUNE=0` 为纯 DAgger。`TEACHER_CHECKPOINT` 与 `REGISTRY` 的列表顺序必须逐项匹配；本地 `.pt` 教师需要显式提供 registry，不能假设 `REGISTRY=auto` 可解析。
4. **观测：** `wbt_training/config_values/depth_distillation.py` 将 actor 限为深度、本体与离散速度命令，不暴露参考运动、height scan 或根线速度；教师和 critic 的特权观测不可误写成部署输入。学生关闭教师式 adaptive motion timestep sampling。
5. **导出：** 训练 checkpoint 配套导出两个 ONNX；也可用 `python -m wbt_training.training_runs.export_distill_onnx` 手动导出。
6. **闭环推理：** `run_php_sim.sh` → Holosoma `run_sim.py` 发布深度共享内存与机器人状态；`run_php_inference.sh` → `run_policy.py` 用两份 ONNX 输出关节目标，经本地 simulator bridge 回到 MuJoCo。

## 环境与复现注意

- 训练要求 Linux / NVIDIA GPU 与 IsaacSim 5.1；motion matching 有独立环境，不需要 IsaacSim。训练脚本默认 W&B，可用 `LOGGER=disabled` 配合本地文件走无账号路径。
- README 提醒 IsaacSim 5.1 与 Holosoma 的 `typing_extensions` 约束冲突，不能把 `pip check` 的依赖报告等同于已验证运行失败；安装时以官方环境说明为准。
- `--training.num-envs` 是多 GPU 合计环境数；公开示例的 4096 环境并非论文的 16384 环境配置。学生 launcher 的 warmup 参数也应显式记录，不能只凭论文描述推定脚本默认值。
- 仿真先启动，等 `depth_img_shm` 创建后再启动推理；深度形状 `(1, 1, 58, 87)`，20184 bytes。`--simulator.config.sim.fps=500` 是仿真物理频率，不代表策略或相机同频。
- 本地推理传入同次导出的 `BACKBONE` / `STUDENT`；无需 W&B 或 FAR-pi。推理脚本不会自动将 `.pt` 转 ONNX，也不解析 `STEP=latest`。
- 本地需要图形桌面；复用旧 Holosoma editable 安装可能导入错误 checkout。浏览器 demo 的 `Y` 切换速度，与原生推理的 `=` 不同。
- 本次只进行了官方文档与源码核查；没有下载全部权重、安装 IsaacSim 或执行训练 / 真机测试。

## 对 wiki 的映射

- 统一补充现有 [PHP 实体](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)，不创建第二个项目节点。
- 关联 [Holosoma](../../wiki/entities/holosoma.md)、[DAgger](../../wiki/methods/dagger.md) 与 [楼梯 / 障碍感知 locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)。
- 配套归档：[项目页](../sites/php-parkour-github-io.md)、[论文](../papers/php_parkour_arxiv_2602_15827.md)。
