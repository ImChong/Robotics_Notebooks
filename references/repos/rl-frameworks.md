# RL Frameworks

人形/腿足机器人 RL 训练常用开源框架。

## 核心框架

### IsaacGym + IsaacLab

- **IsaacGymEnvs** — NVIDIA GPU 并行 RL 训练底座，适合大规模并行仿真
- **IsaacLab** — IsaacGym 的下一代，支持更复杂场景和人形机器人
- [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)

### legged_gym

- 四足/人形腿式机器人 RL 训练框架，基于 IsaacGym
- 学走路、跑、越障的标准起点
- [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)

### humanoid-gym

- 人形机器人 RL 专用，基于 IsaacGym
- 聚焦双足步行、站起、扰动恢复

### MimicKit

- 模仿学习 + 技能迁移框架
- 支持 MoCap 数据导入和重定向
- [Stanford HAI Lab](https://motion.stanford.edu/research/mimickit)

### MotrixLab

- **MotrixLab** — 基于 MotrixSim (Rust) 的通用机器人训练平台
- 支持 JAX 和 PyTorch，兼容 MJCF
- [Motphys/MotrixLab](https://github.com/Motphys/MotrixLab)

## 关联页面

- [[wiki/entities/isaac-gym-isaac-lab]] — Isaac Gym 实体页
- [[wiki/entities/legged-gym]] — legged_gym 实体页
- [[wiki/entities/motrix]] — Motrix 实体页
- [[references/repos/simulation]] — 仿真平台层
