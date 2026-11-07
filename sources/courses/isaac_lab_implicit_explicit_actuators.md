# Isaac Lab / mjlab：Implicit vs Explicit Actuator（执行器建模）

> 来源归档（ingest）

- **标题：** 机器人 RL 仿真中 Implicit / Explicit 执行器模型 — Isaac Lab 与 mjlab 官方文档
- **类型：** course（官方文档 / 框架概念）
- **入库日期：** 2026-07-09（2026-11-07 重新读取当时最新文档与源码复核）
- **版本：** Isaac Lab 文档 `main`（页脚 Last updated Nov 07, 2026；页面已迁到 `/source/concepts/actuators.html`，旧 `/overview/core-concepts/actuators.html` 仅为跳转页）；mjlab 文档 `main`（版本列表含 v1.6.0）；Peng & van de Panne arXiv:1611.01055v1。
- **核对范围：** 通读 Isaac Lab Actuators 概念页全文（模型选择表、参数参考、Implicit vs explicit、执行器流水线、属性归属、原生执行器、BAM 伺服模型、后端提交）；读 `source/isaaclab/isaaclab/actuators/` 下 `actuator_pd.py`（ImplicitActuator / IdealPDActuator / DCMotor 的类说明与 clip 逻辑）与 `actuator_net.py`（ActuatorNetMLP / LSTM 的历史缓冲与缩放）的相关片段，其余文件与 API 参考页未读；通读 mjlab Actuators 概念页（内置 vs 显式、执行器类型、延迟、配置示例、由电机参数计算 armature 与 PD 增益）至「自定义执行器」小节开头，自定义执行器扩展接口部分未读；通读 Peng 论文正文（动作空间对比实验、鲁棒性与查询频率），补充材料表格只浏览。未运行任何代码。
- **通读：** 已完成
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

## 2026-11-07 复核补充（以最新文档为准，旧摘录中与之冲突者见下）

- **Isaac Lab 现行表述：** 两种方式——implicit 把位置 / 速度增益交给物理引擎，在求解器内运行弹簧阻尼（PD）；explicit 运行一个计算关节力的模型、按电机能力限幅后提交（可建模饱和、延迟、齿轮与学习到的电机行为）。模型表：`ImplicitActuator`、`IdealPDActuator`、`DCMotor`、`DelayedPDActuator`、`RemotizedPDActuator`、`ActuatorNetMLP / ActuatorNetLSTM`、`BamActuatorCfg`（仅 Newton）。**Isaac Lab 里没有 `LearnedMlpActuator` / `DcMotorActuator` 这两个名字**——那是 mjlab 的命名。
- **迁移与稳定性：** 现行文档称 implicit 与 ideal-PD explicit 在增益相同、dt=1/360 s 的单关节示例中响应几乎一致，但示例之外可能不同，「用 implicit 训的策略换到 explicit 可能需要调整」；explicit 发散时建议检查质量惯量、目标缩放与更新率、力矩限幅、增益、时间步与求解器收敛，armature 应依电机与传动模型选取。旧版摘录里的「PhysX 对期望力矩加数值阻尼」在现行 main 页未再出现。
- **增益归属：** explicit 组的增益由执行器模型持有，求解器自身的 PD 增益置零，所以 `data.joint_stiffness / joint_damping` 对 explicit 关节读到 0；explicit 组仍保留求解器的 `joint_effort_limit`，故模型输出会被二次截断，需让 `joint_effort_limit` 不小于 `actuator_effort_limit`。
- **DCMotor：** 在理想 PD 力矩上套四象限转矩–转速包络，`saturation_effort` 为堵转力矩、`actuator_velocity_limit` 为空载转速；`ActuatorNetMLP/LSTM` 用关节位置误差与速度历史预测力矩，输出再被 DC 电机包络截断，需 TorchScript 检查点；代码注释写明 MLP 版实现 Hwangbo 2019，LSTM 版取自 Rudin 2022 并省去历史缓冲。
- **延迟：** `DelayedPDActuator` 在 reset 时从 [min_delay, max_delay] 物理步均匀采样延迟（示例 0–48 步，dt=1/360 s 对应 0–133 ms）；原生路径下 delay 固定，`min_delay` 被忽略（BAM 例外）。
- **mjlab：** 内置执行器（`BuiltinPositionActuator`、`BuiltinPdActuator` 等）创建原生 MuJoCo actuator，由引擎隐式积分速度相关阻尼；显式执行器（`IdealPdActuator`、`DcMotorActuator`、`LearnedMlpActuator`）在用户代码算力矩，经 `<motor>` 透传，积分器看不到其速度导数，数值上较不稳健；默认 `implicitfast` 积分器。延迟字段（`delay_min_lag` / `delay_max_lag` 等）可加在任意执行器配置上，且与观测延迟区分。
- **增益的经验推导（mjlab 示例）：** $K_p=J_{arm}\omega_n^2$、$K_d=2\zeta J_{arm}\omega_n$，示例取 10 Hz 带宽、$\zeta=2$（过阻尼），因为只算了电机反射惯量而低估总惯量；反射惯量 $J=J_{motor}(N_1N_2)^2+J_{carrier}N_2^2+J_{output}$。
- **Peng & van de Panne（arXiv:1611.01055）：** 比较力矩（Tor）、肌腱激活（MTU）、目标关节角 PD、目标关节速度（Vel）四种动作空间；PD 在全部运动上表现接近最优，且在 7 条运动中的 5 条学得最快，Tor 最终性能普遍最差（角色越复杂差距越大）；PD / Vel 更能适应更低的策略查询频率（测试 15 / 30 / 60 / 120 Hz，主实验 60 Hz）；仅在 2D 平面角色上验证，3D 留作未来工作；PD 增益为人工指定。
