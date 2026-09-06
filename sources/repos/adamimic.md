# InternRobotics/AdaMimic

- **标题：** AdaMimic 官方实现（Adaptive Motion Tracking）
- **类型：** repo
- **URL：** <https://github.com/InternRobotics/AdaMimic>
- **许可：** CC BY-NC-SA 4.0（禁止商业使用）
- **配套论文：** [arXiv:2510.14454](https://arxiv.org/abs/2510.14454) — [`sources/papers/adamimic_arxiv_2510_14454.md`](../papers/adamimic_arxiv_2510_14454.md)
- **项目页：** <https://taohuang13.github.io/adamimic.github.io/>
- **入库日期：** 2026-09-06

## 一句话说明

基于 **Isaac Gym + legged_gym + rsl_rl (PPO)** 的 **两阶段**自适应 motion tracking：Stage1 固定相位关键帧跟踪，Stage2 训练 phase/tracking adapters；`g1_dof27` 多任务配置；`train.py` / `play.py` Hydra 入口。

## 仓库状态（2026-09-06 核查）

| 项 | 内容 |
|----|------|
| 环境 | `conda env create -f conda_env.yml` → `conda activate adamimic`；另需手动安装 Isaac Gym |
| Stage1 训练 | `python legged_gym/scripts/train.py +dataset=g1_dof27/${task} +algorithm=adamimic/stage1` |
| Stage2 训练 | `... +algorithm=adamimic/stage2 checkpoint_path=${stage1_ckpt}` |
| 推理 | `python legged_gym/scripts/play.py +dataset=g1_dof27/${task} +algorithm=adamimic/stage2 resume_path=${stage2_ckpt}` |
| 基线 | `+algorithm=${baseline}`（`legged_gym/legged_gym/configs/algorithm/`） |
| 任务列表 | `legged_gym/legged_gym/configs/dataset/g1_dof27/` |
| 栈依赖 | legged_gym、rsl_rl、ASAP/AMP for hardware 等（见 README Acknowledgments） |

**结论：** **已开源、可运行** 训练与 play 管线；真机部署脚本以仓库后续更新为准。

## 与 wiki 的关系

- 实体页：[paper-adamimic](../../wiki/entities/paper-adamimic.md) — 含源码运行时序图。
