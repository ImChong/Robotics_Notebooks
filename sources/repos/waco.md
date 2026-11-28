# peacelwh/WACO：位姿鲁棒 VLA 与基准实现 / Pose-Robust VLA and Benchmark Implementation

- **类型：** repo
- **仓库：** [peacelwh/WACO](https://github.com/peacelwh/WACO)
- **核对版本：** [fcf2dc8](https://github.com/peacelwh/WACO/tree/fcf2dc8849c0c10202f5d5f4a9e23d03e096cdd4)，2026-11-28 核对
- **许可：** 代码 Apache-2.0；权重另循 Gemma 使用条款
- **论文：** [arXiv:2611.24360v1](https://arxiv.org/abs/2611.24360v1)
- **项目页：** [WACO](https://peacelwh.github.io/WACO/)
- **配套：** [项目页归档](../sites/waco-project.md) · [论文归档](../papers/waco_arxiv_2611_24360.md) · [基准资产](../datasets/libero-poseshift.md) · [权重](../models/waco-pi05-libero.md)

| 入口 | 已核对的职责 |
| --- | --- |
| `scripts/install_plugin.sh` | 注册 LeRobot 环境和训练选项；评测使用 LeRobot 0.6.1 |
| `scripts/download_assets.sh` | 默认拉校准和索引，`WITH_DEPTH=1` 额外拉 8.4 GB 级逐帧深度 |
| `scripts/eval_grid.sh` | 扫 suite、扰动类型、强度和种子；可断点续跑；每格调用 `lerobot-eval`，最后聚合结果 |
| `scripts/eval_liberoplus_camera.sh` | 在单独 LIBERO-Plus 环境执行相机轨，根据 `PROMPT` 选择发布或纠正提示 |
| `src/poseshift/libero_env.py` | `LiberoPoseShiftEnv.reset` 载入标定、注入扰动、恢复初态和 settle，调用运行时估计及可选 homing；`step` 执行逆变换 |
| `src/poseshift/estimator.py`、`waco_runtime.py` | 双锚点、门控、图像与状态 canonicalization；后者在策略外反变换 7 维增量动作 |
| `scripts/train_pi05.sh`、`src/poseshift/rt_aug.py` | 在数据读取时开启往返回投影与结构化状态噪声，并保留动作标签 |

公开发布重点是仿真评测和微调，不把论文真机控制栈或 UR5e 示范视为已开放。复现核对跑了仓库轻量测试：24 通过、1 跳过；没有下载完整权重、执行 GPU 仿真闭环或重新训练。

- **对 wiki 的映射：** [WACO](../../wiki/entities/paper-waco-pose-robust-vla.md)
