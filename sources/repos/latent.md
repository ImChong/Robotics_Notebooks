# LATENT

> 来源归档（国内具身开源全景 · 源码核对）

- **标题：** LATENT
- **类型：** repo
- **机构：** 银河通用（GALBOT · 清华大学）
- **链接：** <https://github.com/GalaxyGeneralRobotics/LATENT>
- **论文与项目页：** [arXiv:2603.12686](https://arxiv.org/abs/2603.12686)（IROS 2026 最佳娱乐与游乐论文）· <https://zzk273.github.io/LATENT/>
- **分类：** 全身动作跟踪/技能训练
- **入库日期：** 2026-09-06
- **核查日期：** 2026-11-21
- **一句话说明：** LATENT（人形网球全身技能学习）官方代码仓库：目前公开了基于 OpenTrack/Brax 的网球动作跟踪环境、ONNX 导出与 MuJoCo 渲染评测脚本及小规模网球动捕子集；DAgger 隐空间蒸馏、高层网球策略训练与真机部署代码仍在 TODO 待发布状态。
- **核对范围：** 通读 `GalaxyGeneralRobotics/LATENT@a931da5` 的 `README.md`、`latent_mj/` 全部目录结构、`latent_mj/learning/policy/ppo/train_tracking.py`、`latent_mj/app/brax2onnx_tracking.py` 与 `latent_mj/eval/tracking/mj_onnx_video.py`；未在本机下载网球动捕数据或启动训练/渲染。
- **通读：** 已完成

## 开源状态

- **部分开源**：
  - **已发布（2026-03-13）：** 基于 OpenTrack 的 MuJoCo-MJX 网球动作跟踪环境（`latent_mj/envs/g1_tracking/train/g1_env_tracking_tennis*.py`、`play/play_g1_env_tracking_tennis.py`）、`train_tracking.py`、ONNX 导出（`latent_mj/app/brax2onnx_tracking.py`）与 MuJoCo 推理评估（`latent_mj/eval/tracking/mj_onnx_video.py`），以及一小部分 Google Drive 网球重定向数据。
  - **未发布（README TODO）：** 完整 5 小时网球动作数据、预训练动作跟踪权重、DAgger 在线隐空间蒸馏代码、预训练隐动作模型、高层网球对打训练代码及真机 Sim2Real 部署代码。
  - **入口核对提示：** `README.md` 所列训练命令行 `python -m latent_mj.learning.train.train_ppo_track_tennis` 在 `a931da5` 树中**不存在**（仓库内无 `latent_mj/learning/train/` 目录，仅保留 `latent_mj/learning/policy/ppo/train_tracking.py`、`brax2onnx_tracking.py` 与 `mj_onnx_video.py`）。

## 对 wiki 的映射

- [wiki/entities/paper-notebook-latent.md](../../wiki/entities/paper-notebook-latent.md)
