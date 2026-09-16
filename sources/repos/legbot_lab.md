# Legbot Lab（四足 Isaac Lab RL 训练与 Sim2Real 部署）

- **标题：** Legbot Lab / legbot_lab
- **类型：** repo
- **仓库：** <https://github.com/Robot-Nav/legbot_lab>
- **许可：** Apache License 2.0
- **硬件：** 自研 **Legbot** 四足（12 DoF，~14 kg，RobStride RS02）；结构与 Unitree Go2 同构
- **默认分支：** `PPO`（纯 PPO + 非对称 Actor–Critic + 10 帧历史）
- **扩展分支：** `PPO-CTS-MOE`（MoE-CTS）；`WF-CTS-MOE`（四轮足 WF-CTS-MOE）
- **收录日期：** 2026-09-16
- **开源结论：** **已开源**（训练、MuJoCo sim2sim、C++ ONNX 部署、串口 DDS 网关齐全）

## 一句话摘要

基于 **NVIDIA Isaac Sim / Isaac Lab** 的 Legbot 四足 **PPO** 训练框架：4096 并行环境、非对称 critic 特权、历史观测、域随机化；**ONNX / TorchScript** 导出后经 **C++17 + CycloneDDS + 串口网关** sim2real；`PPO-CTS-MOE` 分支实现 **并发 Teacher–Student + 8 专家 MoE** 学生编码器。

## 为何值得保留

- **完整 Sim2Real 闭环：** Isaac Lab 训练 → MuJoCo DDS 仿真验证 → 香橙派 ONNX 控制器 → 真机，与 [legbot-MPC-WBC](../../wiki/entities/legbot-mpc-wbc.md) 形成 Robot-Nav **RL / MPC 双开源线**。
- **算法栈清晰：** `PPO` 基线文档化 MDP、奖励、域随机化；`PPO-CTS-MOE` 对齐 [CTS 论文](../papers/legbot_cts_arxiv_2405_10830.md) 并扩展 MoE。
- **工程细节可复现：** 1 kHz FSM 控制器、500 Hz 串口网关、安全限幅与 Passive 急停逻辑均有 README 与 YAML 配置。

## 技术要点（编译自 README）

| 项 | 内容 |
|----|------|
| 仿真 | Isaac Sim 5.0 / Isaac Lab 2.2；PhysX GPU 4096 envs |
| RL | RSL-RL 2.3.1 PPO；clip ε=0.2；GAE γ=0.99 λ=0.95 |
| 观测 | Actor 45×10 帧历史=450 维；Critic 263 维特权 |
| 动作 | 12 维关节位置偏移 → PD（Kp=50 Kd=3 训练） |
| 导出 | `play.py` → `policy.onnx` / `policy.pt` / `deploy.yaml` |
| 部署 | C++17 ONNX Runtime；CycloneDDS `rt/lowcmd`/`rt/lowstate` |
| sim2sim | MuJoCo + DDS bridge（`simulate/`） |
| 真机 IO | `serial_dds_gateway` USB-CAN + 串口 IMU |
| MoE-CTS | 75% teacher / 25% student env；8 experts + gating；latent 蒸馏 |

## 分支对照

| 分支 | 算法 | 典型任务 |
|------|------|----------|
| `PPO` | PPO + 非对称 AC + 历史 | `Unitree-Legbot-Velocity` 速度跟踪 |
| `PPO-CTS-MOE` | MoE-CTS（并发 TS + MoE 学生编码器） | 多地形盲走 |
| `WF-CTS-MOE` | WF-CTS-MOE | 四轮足变体 |

## 对 Wiki 的映射

- [legbot-lab 实体页](../../wiki/entities/legbot-lab.md)
- [paper-cts-concurrent-teacher-student-locomotion](../../wiki/entities/paper-cts-concurrent-teacher-student-locomotion.md)
- [legbot-mpc-wbc](../../wiki/entities/legbot-mpc-wbc.md)（同团队 MPC 线）
- [teacher-student-dagger-training](../../wiki/methods/teacher-student-dagger-training.md)

## 参考来源（原始）

- 代码：<https://github.com/Robot-Nav/legbot_lab>
- 相关论文：[arXiv:2405.10830](https://arxiv.org/abs/2405.10830)（CTS）
- 公众号导读：<https://mp.weixin.qq.com/s/1XB0Dav8vtg52DZJXK1EcA>（入库日可访问性未逐字归档）
