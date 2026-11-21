# X2Real（X-Square-Robot/x2real）

> 来源归档（repo / 官方评测与手动采数运行时）

- **标题：** X2Real — An eXtensive simulation benchmark for real-world generalist policies
- **类型：** repo
- **机构：** 自变量机器人（X Square Robot）
- **链接：** <https://github.com/X-Square-Robot/x2real>
- **许可：** Apache-2.0（根目录 `LICENSE`；第三方与独立分发的资产各自适用许可）
- **版本：** X2Real **v0.9.0**（2026-10-10 发布）
- **配套论文：** [arXiv:2609.27449](https://arxiv.org/abs/2609.27449)
- **项目页：** <https://x2robot.com/en/pages/x2real>
- **资产与数据集：** <https://huggingface.co/datasets/x-square-robot/x2real-assets>；<https://huggingface.co/datasets/x-square-robot/X2Real-Dataset>
- **入库日期：** 2026-11-21
- **核对范围：** 检出 `X-Square-Robot/x2real@7ed53b4fac4f9836475a7aced38f64ad123b3c73`（main），通读 `README.md`、`OPEN_SOURCE_SCOPE.md`、`manaenv/scripts/x2real_eval.py` 与 `manaenv/scripts/x2real_collect.py`；未运行 NVIDIA GPU + Isaac Sim 6.0.1 闭环仿真
- **通读：** 已完成

## 仓库定位与开源范围（`OPEN_SOURCE_SCOPE.md`）

- **已开源（Apache-2.0）：**
  - **ID 评测运行时（`manaenv/scripts/x2real_eval.py` → `manaenv/scripts/main.py`）：** 支持 **5 个正式 ID 评测板、73 个可运行 ID case**（`artixon_arm6a_tabletop` 38、`artixon_arm6a_challenge` 10、`quanta_x1_mobile` 3、`arx_r5_tabletop` 12、`franka_panda_tabletop` 10），默认每 case 40 episode / 4 并行环境，通过 `--resume` 支持断点续跑，输出 `benchmark_report.json` 与录制视频。
  - **Policy Space v2 协议端（`3rd/policy_space`）：** `policy-space check` / `policy-space serve` 将策略服务与 ManaEnv 仿真进程解耦。
  - **手动示教采数（`manaenv/scripts/x2real_collect.py`）：** 支持 ArtiXon Arm-6A 与 Quanta X1 的键盘、Vuer/WebXR、WebSocket 主从臂（`masterslave`）及直接 IK/关节控制，输出本地 HDF5。
- **未随仓发布（明确排除范围）：** 自动规划采数（automatic planner collection）、CuRobo 规划、回放（replay）、Mimic、数据集清洗、LeRobot 格式转换、训练脚本、**OOD 评测定义（`--ood` 等参数被公共入口显式禁用）**、通用导航/ROS 栈及私有硬件端点；仿真静态资产与轨迹数据集分别托管于 Hugging Face。

## 对 wiki 的映射

- [X2Real 论文详情](../../wiki/entities/paper-x2real.md)
- [X2Real 项目详情](../../wiki/entities/x2real-project.md)
