# ExcitedButter/tract

- **地址：** <https://github.com/ExcitedButter/tract>
- **固定提交：** `8bd06abbc0d17e8094cd15dcec998bf131a9d44c`（2026-11-13核查）。
- **核对范围：** README全文；`scripts/evaluate_full_track.py`的配置、环境循环和selector构建；`libero_action_selection.py`、`openpi_policy.py`、`vlac_dualview_critic.py`；`third_party/track_dualview/inference.py`。仅静态源码阅读，未训练/执行。
- **入口：** `xvfb-run -a python scripts/evaluate_full_track.py --suites libero_integral --all-tasks`；需自行提供policy、TWM、VLAC等检查点和LIBERO资产。README有训练/标注命令，但不提供项目checkpoint和数据；原中间gripper数据及LIBERO WM loader缺失，不能声称一键复现论文表。
- **调用：** 环境准备观测→`LiberoActionSelector.select_best_candidate`→OpenPI批采样动作/track→TrackDualViewWM批量想象→VLAC双视角加权→argmax→环境执行动作前缀；候选全部评分失败回退0，需要检查日志和HDF5分数。
- **版本边界：** 当前默认replan_steps=5，论文完整执行16；代码TSR k5/noise3/Sobol，论文仿真k3/noise2；README30K训练配置不同于论文微调5K。论文critic范围0–50，仓双视角接口写0–100且默认.6/.4加权，不混用延迟和分数口径。
- **源码细节：** OpenPI关闭include_track_targets以避免需要未来真值；TWM将两个视角沿高度拼接、track渲染heatmap后VAE编码，ControlNet条件化SVD。当前wrapper可条件化文本，README称所报配置无文本，复现须确认实际配置。
- **许可：** openpi派生代码Apache2.0，Gemma权重及third_party各有独立条款；不能把整栈视作单一数据许可。
- **映射：** [TrAct](../../wiki/entities/paper-tract.md)。
