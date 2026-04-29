# Motrix (Motphys 机器人仿真与训练平台)

- **标题**: MotrixSim / MotrixLab
- **链接**: 
  - https://github.com/Motphys/MotrixLab
  - https://github.com/Motphys/motrixsim-docs
- **类型**: repo / simulation-engine / rl-platform
- **作者**: Motphys
- **核心关注点**: 高性能多体动力学仿真、Rust 后端、MJCF 兼容性、强化学习集成

## 核心内容摘要

### 1. 平台组成
- **MotrixSim**: 核心物理仿真引擎。定位为工业级高性能引擎，专注于多体动力学，强调计算精度与效率。
- **MotrixLab**: 建立在 MotrixSim 之上的强化学习（RL）平台。提供统一的机器人训练接口，将仿真环境与现代 RL 框架（如 SKRL, RSLRL）深度集成。

### 2. 关键技术特性
- **高性能并行**:
  - 核心引擎使用 **Rust** 编写（CPU 版本），保证内存安全与极致性能。
  - 针对高吞吐量 RL 训练进行了优化。
  - 深度集成 **JAX** 和 **PyTorch**，支持高效的数据交换。
- **物理建模与求解**:
  - 采用 **广义坐标 (Generalized Coordinates)** 建模（类似于 MuJoCo 的 Articulated Body Algorithm），对关节型机器人仿真更精确。
  - 自研约束模型与求解器，旨在提供稳定高效的多体动力学模拟。
  - 高度兼容 **MJCF (MuJoCo)** 模型格式，方便现有资产迁移。
- **机器人支持**:
  - 针对足式机器人（四足、双足）提供了专门的 `legged_gym` 环境。
  - 支持机械臂操作（Manipulation）任务及复杂交互。

### 3. 与其他引擎的对比
- **vs. MuJoCo**: MotrixSim 是其直接竞争者，同样采用广义坐标和 MJCF 格式，但通过 Rust 实现现代化的性能优化。
- **vs. Isaac Lab**: 不同于极度依赖 NVIDIA GPU 的 Isaac 系列，Motrix 强调高性能的 **CPU 后端**，在 GPU 资源受限或需要 CPU 确定性的场景下更具优势。

### 4. 应用场景
- **运动控制**: 开发与测试传统控制算法（MPC, WBC）。
- **强化学习**: 训练腿式行走与机械臂操作。
- **工业仿真**: 数字化工厂与数字孪生。

## 对 Wiki 的映射
- **wiki/entities/motrix.md** (新建)
- **references/repos/simulation.md** (更新)
- **references/repos/rl-frameworks.md** (更新)
