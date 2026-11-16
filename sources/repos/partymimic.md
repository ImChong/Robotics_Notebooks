# PartyMimic — 训练、数据及评测源码

> 官方代码与工程文档归档；记录可执行入口与许可边界，不镜像仓库全文。

- **标题：** RoboParty Lab / PartyMimic
- **类型：** repo（可微世界模型全身运动跟踪训练与 sim2sim 评测）
- **官方代码：** https://github.com/Roboparty/PartyMimic
- **项目主页：** https://roboparty.github.io/PartyMimic/
- **许可证：** 仓库代码 MIT；数据集与其衍生 checkpoint、视频等遵守各自来源许可，**不可当作全部 MIT**。
- **核对日期：** 2026-11-11
- **核对范围：** 通读官方 README 全文（概述、对比表、权重、安装、四个数据集与重定向、训练、稀疏输入、导出/评测、速度、配方及许可证）；按运行入口阅读 `scripts/train.sh`、`scripts/eval.sh`、`src/partymimic/train.py` 关键初始化/收集/世界模型更新/策略与 critic 更新/主循环、`src/partymimic/env.py`、`networks.py`、`motion.py`、`objective.py`、`optim.py`、`tools/export_deploy_onnx.py` 的相关入口与函数；不声称逐行通读仓库全部源码。
- **通读：** 已完成
- **开源状态：** 训练、数据管线、评测、导出及两份 MLP checkpoint 已公开；BONES-SEED 下载有门槛。
- **沉淀到 wiki：** [PartyMimic](../../wiki/entities/partymimic.md)
- **配套主页归档：** [项目页原始资料](../sites/partymimic.md)

## 经源码核验的运行入口

| 环节 | 文件 / 入口 | 作用 |
|---|---|---|
| 依赖 | `uv sync`、`tools/download_data.sh` | CUDA 12.8 环境及基础 robot/motion 资源 |
| 组装数据 | `tools/data/build_library.sh g1` | LAFAN1 + BONES-SEED 10k + KungfuAthlete 训练/Extreme 测试 |
| RP1 重定向 | `tools/data/build_library.sh rp1` | 调用 human-humanoid-tools 的 G1→RP1 IK；需要外部 checkout |
| 训练 | `bash scripts/train.sh logs/g1 --robot g1mj --motion_dir data/motions/g1_mix` | 实际调用 `python -m partymimic.train` |
| 经验数据采集 | `train.py::collect`、`env.py::TrackingSim`、`buffer.py::ReplayBuffer` | 从 mjlab / MuJoCo-Warp 模拟器写入状态转移 |
| 模型与策略学习 | `train.py::wm_update`、`policy_step`、`critic_fit` | WM 拟合 → 32 步想象 BPTT → Critic GAE 拟合 |
| 目标函数 | `objective.py::actor_objective` | 折扣跟踪奖励、存活概率与终端价值 bootstrap |
| 导出 | `tools/export_deploy_onnx.py` | 导出 checkpoint → ONNX 与部署 YAML，`sparse5` 用 `ref_kp` |
| 评测 | `scripts/eval.sh` | ONNX → 外部 EGalahad/sim2real MuJoCo 测试；先应用 `tools/sim2real.patch` |

## 可复现性与许可

- README 要求 Linux x86_64、NVIDIA GPU/CUDA 12.8、`uv`；默认 `--num_envs 4096`、`--iterations 12000`、`--horizon 32`、`--wm_opt muon`、`--pi_opt muon`。
- `--pipeline` 使用两卡：cuda:0 收集并拟合世界模型，cuda:1 用上一轮模型/数据更新 policy 和 critic；为选项而非默认。
- 默认训练是 MLP/`joint`；`--ref_enc xattn` 及 `--interface sparse5` 是独立可选分支，后者附加 keypoint-to-joint 与辅助关节角损失。
- 测试使用 MotionDecode（非商用条款）和 KungfuAthlete Extreme；LAFAN1 数据 CC BY-NC-ND 4.0，BONES-SEED 另有授权限制，不能将权重/动作集的再分发自动视为 MIT。
- 本轮仅做静态源码核查，没有在本环境使用 4090 复测训练或运行 sim2sim。

## 一手链接

- [README](https://github.com/Roboparty/PartyMimic/blob/main/README.md)
- [训练主程序](https://github.com/Roboparty/PartyMimic/blob/main/src/partymimic/train.py)
- [世界模型与策略网络](https://github.com/Roboparty/PartyMimic/blob/main/src/partymimic/networks.py)
- [目标函数](https://github.com/Roboparty/PartyMimic/blob/main/src/partymimic/objective.py)
- [ONNX 导出](https://github.com/Roboparty/PartyMimic/blob/main/tools/export_deploy_onnx.py)

