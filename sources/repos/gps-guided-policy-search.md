# Guided Policy Search（cbfinn/gps）

- **类型：** repo / official research code
- **代码：** https://github.com/cbfinn/gps
- **官方文档：** https://rll.berkeley.edu/gps/
- **相关论文：** https://arxiv.org/abs/1504.00702
- **核查入口：** README、docs/index.md、python/gps/gps_main.py、python/gps/algorithm/algorithm_badmm.py（2026-11-09 查阅公开主分支）

## 运行入口及源码关系

- 文档列出 `python python/gps/gps_main.py box2d_pointmass_example`、`python python/gps/gps_main.py mjc_example` 等 CLI 示例。
- `gps_main.py` 的 `GPSMain.run` 周期性取样，收集 `agent.get_samples`，交给 `_take_iteration`。
- `algorithm_badmm.py` 的 `AlgorithmBADMM.iteration` 先 `_update_dynamics`、`_update_step_size`，在内循环更新 policy、dual 及 trajectories。
- 这是 GPS 的**重实现**，且旧版依赖 Python 2.7、旧 Caffe/TensorFlow、早期 MuJoCo/ROS；不等同论文全部真机实验开源。

## 关联页面

- [Deep Visuomotor Policies](../../wiki/entities/paper-deep-visuomotor-policies.md)
