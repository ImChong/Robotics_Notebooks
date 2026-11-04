# TacClaw-Infinity 官方代码

- **仓库：** <https://github.com/tacclaw-infinity/TacClaw-Infinity>
- **项目页：** <https://tacclaw-infinity.github.io/>（[项目页归档](../sites/tacclaw_infinity.md)）
- **论文：** <https://arxiv.org/abs/2610.20320>（[论文归档](../papers/tacclaw_infinity_arxiv_2610_20320.md)）
- **核查日期：** 2026-11-05
- **开放状态：** 部分公开；Tianji、DM-Sparky、RealMan 的回放和运动学评测工具可获取，策略训练及任务级评测代码待发布。

## Tianji 代码路径

以 `robot_executability_checks/tianji/` 为工作目录：`evaluate.sh DATASET OUTPUT --episodes all` 检查虚拟环境并运行 `scripts/evaluate_arm_quality.py`；该脚本通过 `lerobot_v3.open_dataset` 读取 LeRobot v3 数据、通过 `frames.load_scene_config` 读取场景配置，构建 `TianjiUrdfKinematics` 与阈值并调用 `arm_quality.evaluate_dataset`，最后 `write_outputs` 生成 summary、逐帧 CSV、报告和 dashboard。缺失依赖时 shell 入口调用 `setup.sh`。多进程模式按 episode 分区后汇总。

`scripts/replay_lerobot.py` 是单独的 MuJoCo 回放入口；`--dry-run` 建立场景并加载指定轨迹但不打开查看器。README 强调检查坐标系、TCP 和场景注册；回放及运动学评测不能替代策略成功率或真机安全验证。

- **知识映射：** [TacClaw-Infinity 实体](../../wiki/entities/paper-tacclaw-infinity.md)
