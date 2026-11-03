# ViserDex 官方源码

- **仓库：** <https://github.com/leggedrobotics/ViserDex>
- **论文：** <https://arxiv.org/abs/2604.11138>
- **项目页：** <https://rffr.leggedrobotics.com/works/viserdex/>
- **核对版本：** `937d533ff0bdbbe278a5848c2c745a29787c7adb`（2026-11-03）
- **范围：** Isaac Lab 扩展：高斯渲染及增强、关键点估计训练与评测、用于轨迹采集的默认任务策略；README 未声称公开完整论文真机系统。
- **运行入口：** `scripts/rsl_rl/train.py` 训练 rollout 策略；配置 `viserdex/assets/objects.py` 后，`scripts/pose_estimator/train_pose_estimator.py --task Isaac-ViserDex-Repose-Allegro-Estimator-v0` 在线训练关键点估计；另有 eval/test 脚本。
- **数据：** [ViserDexSplats](https://huggingface.co/datasets/leggedrobotics/ViserDexSplats)，CC BY-NC 4.0；代码 BSD-3-Clause。
- **知识节点：** [ViserDex](../../wiki/entities/paper-viserdex.md)
