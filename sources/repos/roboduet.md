# Source: RoboDuet training implementation

- **Title:** RoboDuet: Learning a Cooperative Policy for Whole-body Legged Loco-Manipulation
- **URL:** https://github.com/locomanip-duet/RoboDuet
- **Project page:** https://locomanip-duet.github.io/
- **Paper:** https://arxiv.org/abs/2403.17367
- **License:** MIT (the repository also contains third-party license notices)
- **Default branch:** master
- **核查日期：** 2026-10-11
- **Type:** Official PyTorch training / simulation code

## 仓库职责与入口

该仓库 README 自称论文官方 PyTorch 实现，提供 Isaac Gym 环境下训练和仿真交互入口：

- **环境：** README 要求 Python 3.8（Isaac Gym 兼容要求）并安装 Isaac Gym Preview 4。
- **训练：** `python scripts/auto_train.py --num_envs 4096 --run_name test_roboduet --sim_device cuda:0 --robot go1`；参数支持 `--robot go1` 或 `--robot go2`，可加 `--headless`。
- **仿真操控：** `scripts/play_by_key.py` 可加载指定 logdir 和 checkpoint，用键盘调节移动与机械臂目标。
- **部署关系：** README 把真机代码指向另一个仓库 [RoboDuet_Deployment](https://github.com/locomanip-duet/RoboDuet_Deployment)，此仓库不应与硬件驱动代码混为一谈。

## 许可和复现边界

仓库主 LICENSE 为 MIT，并附带 legged_gym 与 rsl_rl 的许可文件。代码开放不代表第三方仿真器一并开源：安装步骤要求用户自行获取 NVIDIA Isaac Gym Preview 4。README 提供了可运行的训练/播放命令，但复现论文的硬件实验仍需要独立的部署仓库、对应机器人与设备配置。

## 对 wiki 的映射

- 项目实体：[RoboDuet](../../wiki/entities/paper-roboduet-loco-manipulation.md)
- 论文来源：[arXiv 论文归档](../papers/roboduet_arxiv_2403_17367.md)
- 项目页来源：[RoboDuet 项目页](../sites/roboduet.md)
- 真机部署代码：[RoboDuet_Deployment](./roboduet_deployment.md)
