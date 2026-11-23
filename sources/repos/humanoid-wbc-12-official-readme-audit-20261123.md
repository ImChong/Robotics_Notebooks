# 2026 人形机器人 WBC 12 项：官方 GitHub README 核查（2026-11-23）

> 官方技术资料的跨项目审计；非独立论文详情，也不替代每篇论文全文。

- **类型：** repo / 官方原始资料对照
- **核查日期：** 2026-11-23
- **核对范围：** 逐项打开并通读以下 12 个仓库当时默认分支的 README.md（共 12/12），记录 README 中实际列出的训练、评估、遥操作与部署入口、开源边界；未运行代码、未下载数据集/模型、未独立验证硬件实验；各论文算法论证仍参见各自独立论文节点。
- **通读：** 已完成

## 官方原始资料与入口

| 项目 | 官方代码（README） | README 可核对的主线与运行入口 |
|---|---|---|
| HoloMotion | [HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion/blob/master/README.md) | HoloSMPL → motion retargeting → policy training → evaluation；有 `docs/train_motion_tracking.md`、`docs/motion_retargeting.md`、Sim2Sim、ONNX/config 接口；README 明示 v2 sparse-to-whole-body |
| SONIC / GR00T-WBC | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/README.md) | SONIC 模型资源与 ONNX、50 Hz universal-token controller、VR teleoperation；`./deploy.sh`、`gear_sonic/scripts/launch_inference.py`、训练 quick start；README 特别注明完整训练可能需 64+ GPU |
| OpenTrack / Any2Track | [GalaxyGeneralRobotics/OpenTrack](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/main/README.md) | 专家训练、DAgger 学生蒸馏、adapter 与部署流程；`python -m track_mj.learning.train.train_dagger`，支持特定 G1 动作集 |
| BeyondMimic | [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md) | Isaac Lab 2.1 motion tracking 代码、动作转换及注册、`scripts/rsl_rl/train.py` 与评估；README 引导后来者到 mjlab tracking task，不应把两个仓库默认视为完全相同版本 |
| SuperMimic | [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic/blob/main/README.md) | 基于 mjlab 的可微世界模型运动跟踪训练：`src/supermimic/scripts/train.py`，`src/supermimic/scripts/play.py`；README 未证明全部论文资产/真机部署已公开 |
| M3imic / MultiModalWBC | [Renforce-Dynamics/MultiModalWBC](https://github.com/Renforce-Dynamics/MultiModalWBC/blob/main/README.md) | Isaac Lab + RSL-RL，多运动跟踪和多模态模仿任务；`scripts/tools/list_envs.py`、`scripts/rsl_rl/train.py`；README 的 mjlab 版本仍在 TODO |
| UFO | [Roboparty/UFO](https://github.com/Roboparty/UFO/blob/main/README.md) | mjlab 上的 G1/RP1.3 FB、TeCH 无监督 RL，`./run_train.sh` 与 `./run_inference.sh` 分 tracking/reward/goal；应使用当前目录/命令，勿套用老 `humanoidverse` 布局 |
| OMG | [Tsinghua-MARS-Lab/OMG](https://github.com/Tsinghua-MARS-Lab/OMG/blob/main/README.md) | 多模态生成 → ONNX 导出 → tracker → benchmark/deploy，`python -m omg.cli.generation.export_onnx`；区分生成模型与 HoloMotion 跟踪器 |
| TWIST2 | [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2/blob/main/README.md) | 全身遥操作/数据采集、Unitree SDK2/Redis/IsaacGym 环境；提供 `assets/ckpts/twist2_1017_20k.onnx` 测试；README 声明部分 onboard streaming/inference 文档仍在清理 |
| ResMimic | [amazon-far/ResMimic](https://github.com/amazon-far/ResMimic/blob/main/README.md) | 残差策略面向接触载荷下的全身移动操作；`./train.sh <wandb_entity>` 与 `./eval.sh <wandb_run_id> <checkpoint_iter>` |
| OpenHLM | [OpenHLM-project/OpenHLM](https://github.com/OpenHLM-project/OpenHLM/blob/main/README.md) | G1 robot/HuMI 数据采集、OpenPI 模型训练、policy server + robot client 部署；含 GR00T WBC/VR 遥操分组件及权重数据入口 |
| OpenLambda / λ₀ | [LambdaHumanoid/openlambda](https://github.com/LambdaHumanoid/openlambda/blob/main/README.md) | 人类第一视角采集/动作对齐、λ₀ 训练/推理、机器人侧部署指南；`scripts/prepare.py`、`scripts/train.py`；README 2026-11-11 列出独立 [OnboardMimic](https://github.com/LambdaHumanoid/OnboardMimic) 发布 |

## 核对结论

- **开放程度不等价：** 12 个仓库均公开 README/代码入口，但训练资产、checkpoint、真实硬件部署或最终任务数据的可用范围不同，须参照各自 README、LICENSE 和已有 sources/repos 归档。
- **WBC 层级不等价：** SONIC/BeyondMimic 更偏运动跟踪器，SuperMimic 改训练优化，UFO 侧重无监督控制学习，OMG 负责生成参考，OpenHLM/λ₀ 在上层动作策略与数据管线；不能把它们全部称作传统 QP/WBC 求解器。
- **未经复现：** 表格只证明存在官方运行入口，不证明对应命令在本地已跑通，也不证明文中的 5.5 kg、80% 成功率等数字是跨机型通用结果。

## 对 wiki 的映射

- [WBC 开源 12 项比较](../../wiki/comparisons/humanoid-wbc-open-source-12-2026.md)
- [WBC 概念](../../wiki/concepts/whole-body-control.md)
- [BFM 路线](../../roadmap/depth-bfm.md)
