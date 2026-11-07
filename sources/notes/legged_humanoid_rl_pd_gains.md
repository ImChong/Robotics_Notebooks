# 腿足 / 人形 RL 仿真中的关节 Kp、Kd（刚度、阻尼）设置 — 原始资料索引

> 来源归档（ingest）

- **标题：** Legged / Humanoid RL 中 PD 增益（Kp/Kd）与执行器建模原始资料
- **类型：** notes（工程文档与开源实现索引）
- **入库日期：** 2026-05-12（2026-11-07 重读各锚点一手资料复核）
- **版本：** legged_gym master 分支；Isaac Lab main（文档页脚 Nov 07, 2026）；MuJoCo stable XML 参考；mjlab main 文档（2026-11-07 获取）
- **核对范围：** 通读 `legged_gym/envs/base/legged_robot_config.py` 的 `control` / `sim` / `domain_rand` 段、`anymal_c_rough_config.py` 的 `control` 段、`base/legged_robot.py::_compute_torques`、`anymal_c/anymal.py::_compute_torques`；通读 Isaac Lab Actuators 概念页并读 `actuator_pd.py` / `actuator_net.py` 相关类（详见 `sources/courses/isaac_lab_implicit_explicit_actuators.md`）；读 MuJoCo XML 参考中 `actuator/position` 段；读 mjlab Actuators 页（含由电机规格推导增益示例）。未读：MuJoCo `general` 执行器与积分器章节全文、Isaac Lab 其他执行器源码。
- **通读：** 已完成
- **一句话说明：** 汇总 Isaac / legged_gym / MuJoCo 等栈里「刚度–阻尼」参数在代码与文档中的定义位置，供 sim2real 与 RL 策略接口设计对照。
- **沉淀到 wiki：** 是 → [`wiki/queries/legged-humanoid-rl-pd-gain-setting.md`](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)

## 相关论文索引（Kp/Kd / 动作接口）

- [`sources/papers/rl_pd_action_interface_locomotion.md`](../papers/rl_pd_action_interface_locomotion.md) — 人形 Digit、Cassie 双足、四足 sim2real、可变刚度与扭矩控制等 **10 篇** 代表作链接与「对 wiki 的映射」摘录（2026-05-12 ingest）。

## 为什么值得保留

腿足与人形 RL 常见接口是「策略输出关节目标（位置或残差）+ 底层 PD 力矩」。此时仿真里的 **stiffness / damping（即 Kp/Kd 量纲的关节阻抗）** 会改变接触动力学、有效带宽与训练难度；与 `sim.dt`、控制 `decimation`、力矩限幅强耦合，不适合只靠口头经验调参。

## 一手实现与文档锚点

### 1) legged_gym（Isaac Gym）：`control.stiffness` / `control.damping`

- **基类默认字段说明（含单位注释）：**  
  <https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py>  
  摘录要点：`stiffness` 单位为 **N·m/rad**，`damping` 为 **N·m·s/rad**；`control_type` 可为 `P`（位置）、`V`（速度）、`T`（力矩）；`decimation` 表示每个策略步内、以仿真步长为基准的底层控制更新次数。
- **ANYmal-C 粗糙地形任务上的数值示例：**  
  <https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py>  
  摘录要点：按关节组名配置 `stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}`、`damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}`；同一文件展示 **ActuatorNet** 与 **解析 PD** 二选一（`use_actuator_network`）时的力矩计算路径入口（见同仓库 `anymal.py` 中 `_compute_torques`）。

### 2) Isaac Lab：理想 PD 力矩模型中的 stiffness / damping

- **源码（含数学说明）：**  
  <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/actuators/actuator_pd.py>  
  摘录要点：`IdealPDActuator` 使用  
  \(\tau = k_p (q_{des}-q) + k_d (\dot q_{des}-\dot q) + \tau_{ff}\)，  
  其中 `stiffness`、`damping` 配置项即关节空间 Kp/Kd；`ImplicitActuator` 将 PD 交给物理引擎隐式积分，文档注释强调在大仿真步长下与显式模型的精度差异。

### 3) MuJoCo XML：`<position>` 执行器的 `kp` 属性

- **官方 XML 参考（general 与 position 执行器）：**  
  <https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general>  
  <https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position>  
  摘录要点：位置型执行器通过 `kp`（及力限等）定义从控制输入到力矩的映射；与 RL 中「策略输出 → 执行器」链路直接对应，需与 `timestep`、接触求解器设置一并阅读。

### 4) 域随机化经验（与本主题衔接）

- 本仓库已有 checklist 将 **PD 增益随机化** 列为 sim2real 相关项（Kp/Kd ±20% 量级示例），见 [`wiki/queries/sim2real-checklist.md`](../../wiki/queries/sim2real-checklist.md) 对应条目。

## 对 wiki 的映射

- [`wiki/queries/legged-humanoid-rl-pd-gain-setting.md`](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md) — 将上述锚点整理为可执行的增益设置流程（含 Mermaid 主干图）

## 当前提炼状态

- [x] 已建立开源实现与文档 URL 索引
- [~] 后续可按具体机型（Unitree、Booster、Agility 等）增补厂商白皮书中的关节阻抗表

## 2026-11-07 一手核对补充

- **legged_gym 基类（`legged_robot_config.py`）：** `control_type = 'P'`（P 位置 / V 速度 / T 力矩）；示例 `stiffness = {'joint_a': 10, 'joint_b': 15}` N·m/rad，`damping = {'joint_a': 1.0, 'joint_b': 1.5}` N·m·s/rad；`action_scale = 0.5`（目标角 = action_scale × action + default）；`decimation = 4`；`sim.dt = 0.005`（即策略 50 Hz）；`domain_rand` 中 `randomize_friction = True, friction_range = [0.5, 1.25]`、`push_robots = True`。**基类默认并不含 Kp/Kd 随机化项。**
- **`_compute_torques`（P 模式）：** `torques = p_gains * (actions_scaled + default_dof_pos - dof_pos) - d_gains * dof_vel`，随后按 `torque_limits` 截断；V 模式用速度误差，T 模式直接把缩放后的动作当力矩。
- **ANYmal-C rough：** `stiffness = {'HAA': 80, 'HFE': 80, 'KFE': 80}`，`damping = {'HAA': 2, 'HFE': 2, 'KFE': 2}`，`action_scale = 0.5`，`decimation = 4`，且**该配置里 `use_actuator_network = True`**，加载 `resources/actuator_nets/anydrive_v3_lstm.pt`（LSTM，输入为目标角减当前角与关节速度）；关闭该开关才走上述解析 PD。
- **MuJoCo `<position>`：** 等价于 `gainprm = kp`、`biasprm = (0, -kp, -kv)`；`kv` 为执行器阻尼（文档建议配 `implicitfast` 或 `implicit` 积分器）；`dampratio` 以 $2\sqrt{k_p m}$ 为单位（1 为临界阻尼），反射质量 $m=(JM^{-1}J^T)^{-1}$ 在参考构型计算且考虑 armature，但不含被动阻尼与 frictionloss（这两者不可忽略时可能需 <1 的值）。
- **mjlab 示例：** 由电机反射惯量与目标带宽推导增益 $K_p=J\omega_n^2$、$K_d=2\zeta J\omega_n$（示例 10 Hz、$\zeta=2$），G1 示例中髋 / 膝 / 踝分组使用不同 stiffness / damping / effort_limit / armature（髋 180 / 18，膝 150 / 15，踝 40 / 5）。
