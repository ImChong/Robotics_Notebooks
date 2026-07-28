# RoboNaldo_Deploy（OpenDriveLab/RoboNaldo_Deploy）

> 来源归档

- **标题：** RoboNaldo Deploy — Unitree G1 多策略状态机部署
- **类型：** repo（真机 / MuJoCo 部署）
- **来源：** OpenDriveLab
- **链接：** <https://github.com/OpenDriveLab/RoboNaldo_Deploy>
- **克隆：** `https://github.com/OpenDriveLab/RoboNaldo_Deploy.git`
- **配套训练仓：** <https://github.com/OpenDriveLab/RoboNaldo>（归档见 [robonaldo.md](robonaldo.md)）
- **项目页：** <https://opendrivelab.com/RoboNaldo/>
- **许可：** 仓库根目录 **未声明 LICENSE**（GitHub API `license: null`）；以训练仓 MIT 与 README 使用说明为准，复用前自行确认
- **入库日期：** 2026-07-28
- **一句话说明：** RoboNaldo **真机与 MuJoCo 部署**官方仓：FSM 多策略切换（FreeKick / Loco / AMP 等）、机载 LiDAR/相机感知、Xbox 手柄触发；消费训练仓导出的 ONNX。
- **开源状态：** **已开源**（代码公开；README 指向 2026-06 训练+部署发布）。项目页 Code 主链指向训练仓；部署仓由训练仓 README / submodule 引用。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md`](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`OpenDriveLab/RoboNaldo_Deploy`） |
| 默认分支 | `main` |
| Stars / Forks | ~6 / ~1 |
| 描述 | Official deployment repository for RoboNaldo |
| 平台约束 | **29-DoF G1 + 3-DoF 腰**；若装腰固定支架需先解锁；建议去手 |

## 为何值得保留

- **论文真机栈的可运行入口：** FreeKick 模式（547-dim obs、5-frame history、需机载 LiDAR）对齐论文射门部署叙事。
- **感知模块文档齐全：** `onboard/perception/{lidar,camera}`、`PERCEPTION_ARCHITECTURE.md`、tmux bring-up。
- **与训练仓分工清晰：** 训练在 Isaac Lab；本仓只跑已导出策略（MuJoCo sim + 真机）。

## README 入口（归纳）

| 组件 | 说明 |
|------|------|
| 环境 | Conda `robomimic` · Python 3.8；PyTorch 2.3.1 + CUDA 12.1 |
| 依赖 | `pip install -r requirements.txt` + `unitree_sdk2_python` |
| MuJoCo | `deploy_mujoco/deploy_mujoco.py`；任意球用 `--config-name mujoco_freekick` |
| 真机 | `deploy_real/` + `tools/start_tmux_layout.sh`；文档 `onboard/docs/` |
| FreeKick | 手柄 **R1**；需机载 LiDAR |
| 运行时偏置 | L1/L2 + D-pad 调节目标/球 Y 偏置（±1.5 m） |
| 视频说明 | <https://www.youtube.com/watch?v=BuHNzqebIqc> |

### 策略模式（节选）

| Mode | 触发 | 说明 |
|------|------|------|
| PassiveMode | F1 / L2 release | 阻尼保护 |
| LocoMode | B | 稳定行走 |
| FreeKick | R1 | 射门策略（论文主路径） |
| AMP | A | AMP locomotion |
| BeyondMimicMJ / StandUpMJ | D-pad | 躺卧/起立模仿 |

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 论文实体（主） | [`wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md`](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md) |
| 训练仓 | [`robonaldo.md`](robonaldo.md) |
| 硬件 | [`wiki/entities/unitree-g1.md`](../../wiki/entities/unitree-g1.md) |
| 项目页 | [`sources/sites/opendrivelab-robonaldo.md`](../sites/opendrivelab-robonaldo.md) |

## 参考链接

- 部署仓：<https://github.com/OpenDriveLab/RoboNaldo_Deploy>
- 训练仓：<https://github.com/OpenDriveLab/RoboNaldo>
- 项目页：<https://opendrivelab.com/RoboNaldo/>
- Unitree SDK2 Python：<https://github.com/unitreerobotics/unitree_sdk2_python>
