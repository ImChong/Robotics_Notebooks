# robocurve/inspect-robots

> 来源归档

- **标题：** Inspect Robots — evaluation framework for physical AI
- **类型：** repo
- **组织：** Robocurve
- **代码：** <https://github.com/robocurve/inspect-robots>
- **文档：** <https://docs.inspectrobots.org/>
- **LLM 文档：** <https://docs.inspectrobots.org/llms.txt>
- **Stars：** 636（GitHub API 快照，2026-10-04）
- **License：** MIT
- **Python：** 3.10–3.13
- **状态：** alpha；README 提醒 API 仍可能变化，部署前应锁定版本
- **入库日期：** 2026-09-06；复核：2026-10-04
- **最近复核的主仓提交：** d08442a（2026-10-03，公开 rig 文档与 Pi 0.5 示例更新）
- **一句话说明：** **Physical AI 策略评测框架**：把 Task/Scene、Policy 与 Embodiment 解耦，在兼容性预检后执行仿真或真机 rollout，并生成可复核 EvalLog 与 Rerun 记录。
- **沉淀到 wiki：** [Inspect Robots](../../wiki/entities/inspect-robots.md)、[Robocurve](../../wiki/entities/robocurve.md)

## 开源边界与组件关系

| 部分 | 结论 |
|------|------|
| **核心框架** | 已开源，MIT；核心依赖保持轻量（NumPy + 标准库），可视化、仿真与策略集成按需安装 |
| **策略 / 本体适配器** | 插件式接入。主仓包含 agent、CaP-X、ROS、Isaac Sim、XPolicyLab、voice 等插件；YAM 与 Unitree G1 等适配器作为独立包维护 |
| **Unitree G1 适配器** | 独立仓 [inspect-robots-unitree-g1](https://github.com/robocurve/inspect-robots-unitree-g1)，配套 GR00T PolicyServer；详见[单独来源归档](./robocurve_inspect_robots_unitree_g1.md) |
| **Benchmark 任务集** | 框架提供 Task/Scene/Scorer 机制；任务集合可由独立包提供，例如 [WorldEvals](https://github.com/robocurve/worldevals)，安装核心框架不等于安装一套标准任务 |
| **被测模型** | 不随框架分发；策略服务、模型权重、API 依各自许可与环境配置 |
| **可视化** | Rerun 为可选依赖；EvalLog JSON 是主要评测记录，RRD 是可视化录制 |

## 架构摘录

Inspect Robots 将一次评测定义为三个相对独立的部分：

1. **Task**：包含多个 Scene（指令、初始条件、可选 Target）、评测 horizon 与 scorer。
2. **Policy**：读取 Observation 并输出 ActionChunk；可通过插件对接 VLA、LLM agent 或代码策略。
3. **Embodiment**：定义观测/动作空间、控制频率、reset 与安全相关能力，负责连接真实机器人或仿真器。

运行前会检查策略与本体的动作维度、动作语义、必需相机/状态键、控制频率和场景可实现性。不兼容时在运动开始前报错。Controller 可管理 action chunk 等时序逻辑；Approver 可在动作进入本体前 pass、clamp 或 veto。Scorer 读取已记录 TrialRecord，而不是读取易变的实时环境，因此可从保存的记录重新评分。

~~~mermaid
flowchart LR
  task["Task: scenes + horizon + scorer"] --> check{"Policy × Embodiment compatibility"}
  policy["Policy: VLA / LLM / code"] --> check
  body["Embodiment: robot / simulator"] --> check
  check --> rollout["Controller + Approver rollout"]
  rollout --> record["TrialRecord: steps, transcript, latency"]
  record --> score["Scorer / Grader"]
  score --> log["Versioned EvalLog"]
  rollout -. optional .-> rerun["Rerun stream / RRD"]
~~~

## Unitree G1 接入边界（2026-10-04 复核）

G1 不是由核心仓库直接提供的默认 embodiment。Robocurve 的独立 inspect-robots-unitree-g1 包注册 g1_arms 本体和 gr00t 策略，面向**站立状态的 Unitree G1、7-DoF 双臂与 Isaac-GR00T PolicyServer**。该示例将策略输出与机器人表示统一为一个 16 维绝对关节位置契约；适配器把手臂目标写入 rt/arm_sdk，通过 weight joint 混入仍在运行的全身控制器。

**这个边界很关键：** G1 的腿、腰与平衡仍由原有 locomotion controller 管理；适配器提供的是上肢策略评测入口，不代表框架已经接管 G1 全身动作或步行控制。官方仓库 README 还列出相机服务、DDS 网络接口、控制频率与 GR00T checkpoint tag 等实际运行前提。

### 预检能证明什么

插件的 preflight --dry-run 可检查声明的动作维度/语义、相机和状态键，以及可选场景是否可实现；它不会连接电机或 GR00T 服务，也**不能**验证策略输出到底是绝对值还是相对量、夹爪极性、网络可达性或机器人周边间隙。兼容性通过只是接口层检查，不是安全认证或一次成功评测。

### 真机安全与复现要点

- 首次启用 arm-sdk 时，适配器从测量到的当前姿态初始化目标，再渐升控制权重；不要把 weight ramp 替换成直接切换。
- GR00T 服务需使用与适配器相符的输出语义；相对/绝对关节目标弄错可能使目标突变，而且兼容性检查无法识别。
- control_hz 应与 checkpoint 数据帧率相符；发布频率与策略动作频率不同，不能只凭动作维度判断时序兼容。
- 每次首次部署要确认相机朝向、手部极性、网络接口、机械空间和急停；慢速单关节验证后再执行任务。
- G1 结果至少注明 checkpoint、服务版本、action horizon、控制频率、手型、任务场景、成功判据和人工参与方式，否则横向比较容易把部署差异误当作模型差异。

## 主仓近期变化

**2026-10-03 主仓提交**（d08442a）新增公开 rig reference，将 Pi 0.5 serving 脚本纳入仓库，并明确 Pi 0.5 的 16-step action chunk 与 MolmoAct2 默认 30-step horizon 不同。该变化提醒读者：即使同一 policy client 与本体适配器可复用，action horizon 等模型配置仍须与对应服务端对齐。完整历史见[主仓提交](https://github.com/robocurve/inspect-robots/commit/d08442a9d1f43af4658c8d71e02d461e780286e1)。

## 对 wiki 的映射

- [Inspect Robots](../../wiki/entities/inspect-robots.md) — 框架数据模型、评测闭环与 G1 接入边界
- [Unitree G1](../../wiki/entities/unitree-g1.md) — 硬件平台
- [Unitree G1 软件服务栈](../../wiki/entities/unitree-g1-software-stack.md) — SDK2/DDS/运动服务
- [具身评测基准选型闭环](../../wiki/overview/hub-embodied-eval-benchmark.md) — 本框架属于策略执行评测与 sim-real 校准语境
- [XPolicyLab](../../wiki/entities/xpolicylab.md) — VLA 策略统一接入方向
