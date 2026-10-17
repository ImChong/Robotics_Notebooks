# SONIC_unitree_h2（QinnyuLan/SONIC_unitree_h2，TRACE 官方 H2 部署）

- **类型：** GitHub 仓库
- **仓库：** <https://github.com/QinnyuLan/SONIC_unitree_h2>
- **默认分支：** `main`（35 次提交，2026-10-17 页面所见；最近提交日期未显示）
- **源码状态：** 已公开；README 称 H2 检查点位于 `models/sonic_h2_released/`（检查点文件本身未核实）
- **许可证：** 源码 Apache-2.0；权重与第三方组件见 `licenses/README.md`
- **核验日期：** 2026-10-17（仅读取仓库页面，未克隆、未运行）
- **项目页：** [trace-humanoid-transfer-project.md](../sites/trace-humanoid-transfer-project.md)
- **论文：** [trace_humanoid_transfer_arxiv_2610_17999.md](../papers/trace_humanoid_transfer_arxiv_2610_17999.md)
- **知识节点：** [wiki/entities/paper-trace-humanoid-transfer.md](../../wiki/entities/paper-trace-humanoid-transfer.md)

## 仓库说明

README 描述为 "Official Unitree H2 deployment of TRACE: SONIC whole-body control with motion replay, planning, and VR teleoperation"，用统一命令接口覆盖 G1、H2 与 SMPL 参考运动；直接 token 接口预留给后续 VLA / VLN。

## 目录与入口（来自仓库页）

- 目录：`assets/ config/ docs/ licenses/ models/ motions/ requirements/ scripts/ src/ tests/ third_party/ tools/`
- `src/runtime/` C++ 实时策略；`src/interface/` Python 命令生产者；`src/sim/` 仿真器；`tools/validation/verify_release.py` 发布校验
- 流程：克隆含子模块 → `python3 tools/validation/verify_release.py` → 按 `docs/DEPLOYMENT.md` 安装、构建控制器、跑仿真
- 工作流：Sim2Sim（MuJoCo + C++ 控制器 + DDS）、Sim2Real（H2 全身跟踪）、VR 遥操作（PICO 头手追踪，经 XRoboToolkit）
- 运行：启动为站姿，按 `]` 或 `s` 启用策略，`o` 阻尼停止；策略与规划推理用 TensorRT，规划器为独立服务

## 访问与复现边界

install.sh 内容、构建命令、依赖清单、其它脚本入口页面未显示，需读 `docs/DEPLOYMENT.md`；论文中的 LoRA-PPO 适配训练代码是否包含未核实。
