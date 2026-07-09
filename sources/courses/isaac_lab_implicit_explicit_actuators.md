# Isaac Lab / mjlab：Implicit vs Explicit Actuator（执行器建模）

> 来源归档（ingest）

- **标题：** 机器人 RL 仿真中 Implicit / Explicit 执行器模型 — Isaac Lab 与 mjlab 官方文档
- **类型：** course（官方文档 / 框架概念）
- **入库日期：** 2026-07-09
- **一句话说明：** 在腿足/人形 RL 常见「策略输出关节目标 + 底层 PD」栈中，**implicit** 指物理引擎内部积分 PD 并算力矩；**explicit** 指用户侧先算力矩（理想 PD、DC 电机、学习网络等）再写入仿真——二者影响数值稳定性、训练收敛与 Sim2Real 迁移。
- **沉淀到 wiki：** 是 → [wiki/concepts/implicit-explicit-actuator-modeling.md](../../wiki/concepts/implicit-explicit-actuator-modeling.md)

## 为什么值得保留

这是 Isaac Lab / MuJoCo 生态里 **电机控制层** 的基础概念，但常被与「隐式学习」「显式地形表示」等其它 implicit/explicit 用法混淆。官方文档对两类模型的定义、PD 积分差异、以及 **implicit 训练策略换 explicit 模型不一定能直接迁移** 有明确表述，是 Sim2Real 执行器对齐的工程入口。

## 一手资料

### 1) Isaac Lab — Actuators（核心）

- **链接：** <https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html>
- **API 参考：** <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html>
- **核心摘录：**
  - 关节可为 position / velocity / torque 控制；对 position/velocity，引擎内部用 spring-damp (PD) 将用户指令转为关节力矩。
  - **implicit**：理想仿真机制，由 **physics engine** 提供；设 desired position/velocity 后，引擎 **内部** 计算并施加 efforts；PhysX PD 对期望力矩加数值阻尼，更稳定。
  - **explicit**：外部驱动模型，由 **user** 实现；两步：(1) 算期望关节力矩跟踪输入；(2) 按电机能力 clip 后写入仿真。示例：`IdealPDActuator` 实现 $ \tau = k_p(q_{des}-q) + k_d(\dot{q}_{des}-\dot{q}) + \tau_{ff} $ 再限幅。
  - **迁移注意：** policies trained with implicit actuators **may not transfer** to the same robot with explicit actuators；explicit 不收敛时可增大 `armature` 改善稳定性。
- **对 wiki 的映射：**
  - [Implicit / Explicit 执行器建模](../../wiki/concepts/implicit-explicit-actuator-modeling.md)
  - [Armature 建模](../../wiki/concepts/armature-modeling.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)
  - [Isaac Lab 实体](../../wiki/entities/isaac-lab.md)

### 2) mjlab — Actuators（MuJoCo 栈对照）

- **链接：** <https://mujocolab.github.io/mjlab/main/source/actuators.html>
- **核心摘录：**
  - **Built-in（≈ implicit）**：`BuiltinPositionActuator` 等在 MjSpec 创建原生 MuJoCo actuator；引擎 **隐式积分** 速度相关阻尼力，大步长/高增益时更稳。默认 integrator `implicitfast`。
  - **Explicit**：`IdealPdActuator`、`DcMotorActuator`、`LearnedMlpActuator` 在用户代码算力矩，经 `<motor>` passthrough 写入；积分器无法吸收外部力的速度导数项，**数值上不如 built-in 鲁棒**。
  - **LearnedMlpActuator**：用训练 MLP 从关节状态历史预测力矩，继承 DC 电机速度–扭矩限幅；与 [Actuator Network](../../wiki/methods/actuator-network.md) 同属数据驱动 explicit 路线。
  - **Actuator delay**：可在任意 actuator config 上建模指令延迟（与观测延迟区分）。
- **对 wiki 的映射：**
  - [Implicit / Explicit 执行器建模](../../wiki/concepts/implicit-explicit-actuator-modeling.md)
  - [Actuator Network](../../wiki/methods/actuator-network.md)
  - [关节摩擦模型](../../wiki/concepts/joint-friction-models.md)

### 3) 背景：RL 动作接口与 PD 内环（非 implicit/explicit 定义本身，但构成上下文）

- **链接：** [arXiv:1611.01055](https://arxiv.org/abs/1611.01055)（Xue Bin Peng, SCA 2017 — 动作空间对比）
- **要点：** 腿足/角色 locomotion 中「目标关节角 + PD」常比直出扭矩更易学；implicit/explicit 讨论的是 **仿真里这层 PD 由谁算**，不是策略是否「隐式控电机」。
- **对 wiki 的映射：**
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)
  - [DeepRL 动作空间 SCA 2017](../../wiki/entities/paper-deeprl-locomotion-action-space-sca2017.md)
