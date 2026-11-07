# GR00T-WholeBodyControl — 原始资料归档

- **来源**：https://github.com/NVlabs/GR00T-WholeBodyControl
- **类型**：repo
- **机构**：NVIDIA NVlabs（GEAR WBC）
- **归档日期**：2026-05-14
- **再核日期**：2026-07-22
- **入库日期（通读）：** 2026-11-07
- **版本：** `main` 分支 README（News 最新条目 2026-09-14，IsaacTeleop 1.4 与 Jetson AGX Orin 支持）
- **核对范围：** 通读 `main` 分支官方 README 全文（News、GEAR-SONIC 模型卡与三个检查点、C++ 部署命令、VR 遥操作、Kinematic Planner、SONIC 训练快速开始、仓库组成与环境矩阵、MotionBricks、Decoupled WBC、许可）以及官方文档站的 `Training Guide`、`Training Data`（BONES-SEED）、`Decoupled WBC` 与首页；**未读 `gear_sonic` / `gear_sonic_deploy` / `motionbricks` 的源码文件**，未运行训练或部署，GitHub 网页与 API 不可达（star 数沿用 2026-07-22 的旧值，本次未复核）。
- **通读：** 已完成
- **文档站**：https://nvlabs.github.io/GR00T-WholeBodyControl/
- **GitHub**：约 **2.9k** stars（2026-07-22）；**代码** Apache-2.0（主体），**模型权重** 另受 NVIDIA Open Model License 等约束（见仓库 `LICENSE` / `legal/`）

## 一句话说明

**GR00T Whole-Body Control (WBC)** 单仓聚合 **解耦 WBC**（GR00T N1.5 / N1.6 等使用的 RL 下肢 + IK 上肢）、**GEAR-SONIC**（规模化运动跟踪通用人形控制）与 **MotionBricks**（实时潜空间生成式运动）的训练、评测与部署脚本及文档入口。

## 为什么值得保留

- 官方 **Isaac Lab 2.3.2** 对齐徽章与 [GitHub Pages 文档](https://nvlabs.github.io/GR00T-WholeBodyControl/)，覆盖安装、训练、ZMQ/VR 遥操作、VLA 数据链路与真机 C++ 部署
- **GEAR-SONIC**：SONIC 训练代码、HF 权重、C++ 推理、VR 采集与 [在线交互 Demo](https://nvlabs.github.io/GEAR-SONIC/demo.html)（集成 Kimodo 文生运动）
- **MotionBricks**：子目录预览版（交互 G1 Demo、VQVAE/pose/root 权重、合成训练与表示文档），与 [MotionBricks 论文页](https://nvlabs.github.io/motionbricks/) 对应
- 工程注意事项明确：**Git LFS** 拉取大资源；多用途 **独立 venv**（训练 / MuJoCo / 遥操作 / 数据采集）

## 仓库结构（README 摘要）

| 组件 | 职责 |
|------|------|
| `gear_sonic` | SONIC 训练（PPO、数据处理、Hydra 配置） |
| `gear_sonic_deploy` | 真机 C++ 推理栈 |
| `motionbricks` | MotionBricks 预览代码与资源 |

## 关键复现路径（2026-07-22）

1. `download_from_hf.py` 拉 GEAR-SONIC checkpoint（可选 BONES-SEED → `gear_sonic/data_process`）
2. `gear_sonic/train_agent_trl.py`（Isaac Lab）→ `eval_agent_trl.py`（可 `+export_onnx_only=true`）
3. `gear_sonic_deploy/deploy.sh` sim/real（ONNX/TensorRT，50 Hz）
4. Decoupled WBC / MotionBricks / VLA 采集见 [文档站](https://nvlabs.github.io/GR00T-WholeBodyControl/)

## 对 wiki 的映射

1. **[MotionBricks](../../wiki/methods/motionbricks.md)**：生成式运动层与本仓 `motionbricks/` 子项目对应
2. **[motion-bricks.cpp](../../wiki/entities/motion-bricks-cpp.md)**：社区 C++/GGML G1 推理移植（非本仓官方维护）
2. **[SONIC（规模化运动跟踪）](../../wiki/methods/sonic-motion-tracking.md)**：GEAR-SONIC 论文与训练/部署入口（含详细源码时序图）
3. **[GR00T-WholeBodyControl 实体](../../wiki/entities/gr00t-wholebodycontrol.md)**：本仓导航页（含仓级时序图）
4. **[Foundation Policy / GR00T](../../wiki/concepts/foundation-policy.md)**：N1.5 / N1.6 解耦 WBC 与 VLA 工作流文档（站点 tutorials）
5. **[GR00T-VisualSim2Real](../../wiki/entities/gr00t-visual-sim2real.md)**：同品牌下视觉 Sim2Real 另一官方仓库，分工不同

## 引用（GEAR-SONIC，仓库 README）

```bibtex
@article{luo2025sonic,
  title={SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control},
  author={Luo, Zhengyi and others},
  journal={arXiv preprint arXiv:2511.07820},
  year={2025}
}
```

## 通读核对要点（2026-11-07，均出自官方 README / 文档站）

- **三块内容与分工：** Decoupled WBC（"RL 下肢 + IK 上肢"，用于 GR00T N1.5 / N1.6，另见文档 `references/decoupled_wbc`：Docker 环境、`run_g1_control_loop.py`、Pico 遥操作与数据采集栈）；GEAR-SONIC（人形行为基础模型，用运动跟踪作规模化预训练任务，arXiv:2511.07820）；MotionBricks（实时潜空间生成式运动控制，预览版，README 称合成速率可达 15,000 FPS 量级——引用原文截断，未核对完整表述）。
- **SONIC 检查点（模型卡）：** 3 个 Unitree G1 检查点——默认版（10 帧 × 20 ms 的 SMPL 前瞻，约 200 ms）、低延迟遥操作版（4 帧约 80 ms）、v1.1（朝向归一化目标、腕部位姿增广）；都用 SONIC universal-token 控制器，**64 维潜运动 token，控制器 50 Hz**，支持 SMPL、G1 运动参考、遥操作三类输入；部署为 C++ + TensorRT。README 明确"前瞻帧数不等于端到端遥操作延迟"。v1.1 的已测试部署调参：左右踝俯仰电机（索引 4、10）Kp / Kd 缩放 1.5。
- **训练：** 训练指南称 SONIC 用 universal-token 架构控制 G1（29 DoF），多个并行编码器（G1 关节轨迹 / 遥操作 3 点（头 + 双腕）/ SMPL / 可选 SOMA）经 **FSQ** 投影到共享潜 token 空间，单一解码器输出关节动作；PPO + 辅助损失，Isaac Lab（README 徽章 2.3.2）；官方建议 **64+ GPU** 多机训练（单机 8 卡可行但慢）；数据为 BONES-SEED（142,220 条 / 约 288 小时 / 522 名演员，含 G1 MuJoCo 轨迹；过滤约 8.7% 后剩约 130K）；合格指标：`success_rate > 0.97`、`mpjpe_l < 30 mm`、`mpjpe_g < 200 mm`，训练监控 `anchor_pos_err < 0.15 m`、`body_pos_err < 0.10 m`。评测与导出命令：`eval_agent_trl.py`（指标 / 渲染两种模式）、`+export_onnx_only=true` 导出 `*_encoder.onnx` / `*_decoder.onnx`。
- **许可：** 代码 Apache-2.0；模型权重 NVIDIA Open Model License（允许商用但需署名并遵守 Trustworthy AI 条款）。
- **环境：** 不同用途独立 uv venv（`.venv_sim`、`.venv_teleop`、`.venv_data_collection`），训练需单独安装 Isaac Lab；仓库含 Git LFS 大文件，MotionBricks 检查点（约 2.2 GiB）默认跳过。
- **News 时间线：** 2025-11-12 初始发布（仅 Decoupled WBC）→ 2026-02-19 GEAR-SONIC 检查点 / C++ 推理 / VR 遥操作 → 2026-04-10 训练代码与 HF 检查点 → 2026-04-27 MotionBricks 预览 → 2026-05-07 端到端 VLA 工作流（遥操作采集 + Isaac-GR00T N1.7 微调 + SONIC 部署）→ 2026-07-23 SONIC v1.1 → 2026-08-31 逐电机 Kp/Kd 缩放 → 2026-09-14 IsaacTeleop 1.4 / AGX Orin。
