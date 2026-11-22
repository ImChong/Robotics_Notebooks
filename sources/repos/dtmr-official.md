# DTMR 官方开源实现｜nicolejrkim/DTMR

- **官方仓库：** <https://github.com/nicolejrkim/DTMR>
- **代码关联：** [Dense Temporal Motion Retargeting for Legged Robots](https://arxiv.org/abs/2609.38617)
- **项目页 Code 链接：** <https://jaeryeongnicolekim.com/Dense-Temporal-Motion-Retargeting-For-Legged-Robots/>
- **项目文档：** <https://jaeryeongnicolekim.com/DTMR/>
- **状态（2026-11-22 核查）：** 已开源、有可运行的离线重定向/轨迹跟踪入口；Apache-2.0 主仓（部分机器人模型/数据有独立许可），不等于公开论文全部数据和真机控制训练栈。
- **核对范围：** 仓库 README、`pyproject.toml`、`docs/install.md`、`docs/quickstart.md`、`docs/tutorial.md`、`demo_scripts/demo_100style.sh`，以及 `src/dtmr/tracker/cli.py`、`src/dtmr/tracker/controller.py`、`src/dtmr/tracker/env_build.py`、`src/dtmr/env/env.py`、`src/dtmr/tools/run_100style.py` 入口/核心模块。

## 可运行路径（来自 README / 源码）

- **环境：** Linux、NVIDIA GPU、CUDA 12 驱动；Python 3.10；依赖锁定 `mujoco==3.7.0`、`mujoco-warp==3.7.0.1`，`pip install -e .`。
- **一键示例：** `bash demo_scripts/demo_100style.sh Zombie`，依次 `bvh_100style fetch` → `run_100style prepare`（`dtmr.data.keypoint_retarget`）→ `run_100style track` → `run_100style render`。
- **直接入口：** `dtmr --config src/dtmr/examples/unitree_g1_release_warp.yaml --motion_path <robot_ref_50Hz.npz> --output_dir output/clip --no_viz`。`pyproject.toml` 定义控制台命令 `dtmr = dtmr.tracker:main`，经 `tracker/cli.py:main` 载入配置、`build_env` 创建 `TrackingEnvWarp`、`Tracker.run()` 执行 MPC。
- **核心交互：** `Tracker` 以关节 PD 目标 + `rdphi` 为采样动作，`TrackingEnvWarp` 经 Warp kernels 更新相位并在批量 MuJoCo rollout 中计算代价；控制循环输出状态/相位/动作。结束时 `post/dtw.py` 和 `post/phi_tracking.py` 计算误差。
- **输出：** `<timestamp>_log.npz`、`*_dtw_metrics.yaml`、`*_phi_tracking_metrics.yaml`；可选 `render_rollout` 制作 MP4、Brax HTML 播放。
- **注意：** 文档中的 RTX 4090 性能为作者测试（100STYLE 的 Zombie 10 s 为约 11 min，TwoFootJump 10 s 为约 28 min），并非实时在线 MPC 或已在其他显卡上验证。

## 源码边界与许可

仓库提供可辨识的 GPU MPC 运行入口与数据导入器；本次未发现作为本仓库官方入口的完整 BeyondMimic 策略训练和真机部署流程，因此这两部分不能写成已复现。用户应按各数据集许可证、`models/*/LICENSE` 使用资源。

## 对 wiki 的映射

- [DTMR 论文实体（唯一节点）](../../wiki/entities/paper-dense-temporal-motion-retargeting.md)
- [论文原文核查](../papers/dtmr_arxiv_2609_38617.md)
- [项目页](../sites/dense-temporal-retargeting-project.md)
