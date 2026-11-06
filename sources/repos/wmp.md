# WMP 官方源码

- 仓库：<https://github.com/bytedance/WMP>
- 性质：论文复现代码，仓库包含 Dreamer、legged_gym、rsl_rl 和 A1 资源。
- 训练入口：`python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0`。
- 推理可视化：`python legged_gym/scripts/play.py --task=a1_amp --sim_device=cuda:0 --terrain=climb`。
- README 要求 Isaac Gym Preview 3，报告约 23 GB 显存，建议至少 10000 iterations。
- 代码阅读：`legged_gym/scripts/train.py`、`rsl_rl/runners/wmp_runner.py`，快照 c232c115ada4517453ebded5019078ba055456de。
