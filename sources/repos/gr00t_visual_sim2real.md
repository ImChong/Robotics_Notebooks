# GR00T-VisualSim2Real — 原始资料归档

- **来源**：https://github.com/NVlabs/GR00T-VisualSim2Real
- **类型**：repo
- **机构**：NVIDIA Research (NVlabs)
- **归档日期**：2026-05-01（GitHub 元数据更新：2026-05-17）
- **GitHub**：247 stars, 19 forks，Apache-2.0 License

## 一句话说明

GR00T-VisualSim2Real 是 NVIDIA 开源的 **视觉 Sim2Real 框架**，包含两项 CVPR 2026 研究：VIRAL（人形 Loco-Manipulation 规模化视觉迁移）和 DoorMan（像素到动作的策略迁移），在 Isaac Sim 中训练后零样本迁移到 Unitree G1 真机。

## 为什么值得保留

- NVIDIA NVlabs 官方开源，工程质量高
- 两篇 CVPR 2026 论文的配套代码
- 系统展示了 Teacher（PPO 特权状态）→ Student（DAgger RGB 蒸馏）的视觉 Sim2Real 完整 pipeline
- 直接在 Unitree G1 上做 loco-manipulation 零样本迁移，与本知识库多个实体（wbc_fsm、unitree_rl_mjlab、mjlab）高度相关

## 两个子研究

### VIRAL — Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation

- **论文专档（摘录与 wiki 映射）**：[sources/papers/viral-humanoid-visual-sim2real.md](../papers/viral-humanoid-visual-sim2real.md)
- **论文**：arXiv:2511.15200，CVPR 2026
- **项目页**：https://viral-humanoid.github.io/
- **核心问题**：如何让人形机器人在仿真中学到的 loco-manipulation 策略，以仅靠 RGB 相机输入的方式零样本迁移到真机
- **规模**：大量并行环境下 PPO 训练 + DAgger 蒸馏

### DoorMan — Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer

- **论文专档（摘录与 wiki 映射）**：[sources/papers/doorman_opening_sim2real_arxiv_2512_01061.md](../papers/doorman_opening_sim2real_arxiv_2512_01061.md)
- **项目页归档**：[sources/sites/doorman-humanoid-github-io.md](../sites/doorman-humanoid-github-io.md)
- **论文**：arXiv:2512.01061，CVPR 2026
- **项目页**：https://doorman-humanoid.github.io/
- **核心问题**：像素到动作（Pixel-to-Action）的策略迁移，专注于开门任务（重型门，不同阻力）
- **难点**：接触丰富、力反馈难以在视觉观测中隐式推断
- **相对 VIRAL 的增量**：学生阶段后增加 **GRPO** 微调（主信号为任务成功 + 简单行为正则）；教师训练使用 **分阶段重置（staged-reset）快照缓冲** 改善长时域探索；Isaac Lab **程序化门** 强化 **铰链/闩锁/把手** 等物理随机化与 **PBR + dome light** 视觉随机化（论文表 1 消融）。

## 核心技术架构

### Teacher-Student Distillation Pipeline

```
阶段 1：Teacher 训练（Isaac Sim / Isaac Lab，PPO）
  输入：完整特权状态（ground-truth 位姿、物体状态、接触力）
  产出：高性能状态策略 π_teacher

阶段 2：Student 蒸馏（DAgger）
  输入：RGB 相机图像 + 本体感知（关节角、IMU）
  目标：模仿 Teacher 动作分布
  产出：视觉策略 π_student

阶段 3（DoorMan 主路径）：GRPO 微调
  信号：以任务成功二值为主 + 关节/动作平滑等轻量正则
  产出：在 RGB 分布上自举后的 π_student′

阶段 4：部署（Unitree G1 真机）
  导出：ONNX 模型
  输入：RGB + 本体感知
  推理：实时运行
```

### 关键技术要素

| 要素 | 细节 |
|------|------|
| 仿真平台 | Isaac Sim 5.1.0.0 + Isaac Lab |
| RL 算法 | PPO（Teacher）；可选 GRPO（DoorMan 学生自举微调） |
| 蒸馏算法 | DAgger（Student 阶段） |
| 训练框架 | TRL（HuggingFace），Hydra 配置管理 |
| 视觉观测 | RGB 相机（无深度，无 LiDAR） |
| 本体感知 | 关节角、速度、IMU（43-DOF Unitree G1） |
| 部署格式 | ONNX 自动导出 |
| 实验追踪 | Weights & Biases 集成 |
| 域随机化 | Isaac Lab 配置化 DR |

## 项目结构

```
gr00t/rl/
├── train_agent_trl.py      # 训练入口
├── eval_agent_trl.py       # 评估入口（含 ONNX 导出）
├── config/                 # Hydra YAML 配置
├── envs/                   # 任务环境实现
├── trl/                    # Trainer、模块、回调
├── agents/modules/         # 神经网络组件
├── simulator/isaacsim/     # Isaac Sim 接口
└── data/                   # 机器人资产与场景
```

## 依赖与要求

| 依赖 | 版本 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.7.0 |
| Isaac Sim | 5.1.0.0 |
| Ubuntu | 22.04 |
| NVIDIA GPU driver | ≥ 535 |

## 支持任务

- Loco-Manipulation：行走 + 抓取 + 物体操作的组合任务
- Door Opening：开重型门（不同阻力等级）
- Pick-and-Place：取放操作
- 多步骤操作序列

## 关键洞见（对 wiki 的映射）

1. **视觉 Sim2Real 的完整闭环**：从仿真训练到 ONNX 部署的完整工程示范，是 sim2real 概念的典型落地案例
2. **特权状态 Teacher + RGB Student 蒸馏**：PPO + DAgger 组合规避了视觉直接 RL 的样本效率问题
3. **Unitree G1 全身人形任务**：与 wbc_fsm、unitree_rl_mjlab 系列形成"从训练到部署"的完整链条
4. **CVPR 2026 双发**：VIRAL + DoorMan 同时被 CVPR 接收，说明视觉 Sim2Real 正在成为人形机器人操作的主流方向
5. **DoorMan 专档**：[sources/papers/doorman_opening_sim2real_arxiv_2512_01061.md](../papers/doorman_opening_sim2real_arxiv_2512_01061.md) — 分阶段重置、GRPO 与大规模门随机化的论文摘录与 wiki 映射
