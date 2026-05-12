# 腿足 / 人形 RL 仿真中的关节 Kp、Kd（刚度、阻尼）设置 — 原始资料索引

> 来源归档（ingest）

- **标题：** Legged / Humanoid RL 中 PD 增益（Kp/Kd）与执行器建模原始资料
- **类型：** notes（工程文档与开源实现索引）
- **入库日期：** 2026-05-12
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
