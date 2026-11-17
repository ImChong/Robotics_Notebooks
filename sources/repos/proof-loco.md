# syw-robotics/proof-loco 官方训练代码

- **类型：** repo
- **仓库：** https://github.com/syw-robotics/proof-loco
- **论文：** https://arxiv.org/abs/2611.18305
- **项目：** https://syw-robotics.github.io/proof-loco/
- **核对日期：** 2026-11-17
- **开源状态：** 训练代码已发布；部署代码 TODO 未完成。
- **核对范围：** README 安装、注册任务、训练与播放命令；未执行代码，也未逐行审计实现。
- **通读：** 未完成

## 运行入口（README 已核对）

- 教师预热：`python scripts/z_rl/train.py --task Proof-Heightmap-Parkour-PPO-Go2 --physics ovphysx --num_envs 4096 --max_iterations 1500 --pretrain_terrain`
- 教师续训：同入口、同任务，`--max_iterations 8500 --resume`。
- 学生蒸馏：`python scripts/z_rl/train.py --task Proof-Vision-Parkour-Distill-Go2 --physics ovphysx --num_envs 4096 --max_iterations 10000 --teacher_checkpoint /absolute/path/to/teacher/model_99.pt`
- 播放：`python scripts/z_rl/play.py --task Proof-Vision-Parkour-Distill-Go2 --physics ovphysx --num_envs 40 --viser`
- README 建议使用 PhysX，并提示 CUDA/cuDNN 动态 batch 初始化问题。

## 对 wiki 的映射

- [PROOF](../../wiki/entities/paper-proof-loco.md)
