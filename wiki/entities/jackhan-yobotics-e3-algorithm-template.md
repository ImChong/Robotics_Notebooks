---
type: entity
title: Yobotics HumanoidE3 外接算法模板（LCM）
tags: [humanoid, lcm, onnx, pytorch, template, jackhan-sdu, yobotics]
summary: "提供 AlgorithmBase：ONNX 或 PyTorch 加载、LCM 订阅/发布、50Hz 策略线程 + 500Hz 发令线程与锁；开发者实现 compute_observation / process_action 即可接入 WalkE3-Controller 的 DEVELOPMENT 模式。"
updated: 2026-11-13
status: complete
related:
  - ./jackhan-walke3-controller.md
  - ./jackhan-walke3-e3-ecosystem.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/repos/jackhan-algorithm-template-for-developer.md
---

# Yobotics HumanoidE3 外接算法模板（LCM）

本模板用 LCM 与 WalkE3-Controller 的 DEVELOPMENT 模式对接，提供 AlgorithmBase 双线程（约 50Hz 推理与 500Hz 发令）与 ONNX/PyTorch 加载示例。

**定位**：把「**状态机内 RL**」与「**外接 Python 算法**」解耦：模板通过 **LCM** 订阅 `development_state_t`、发布 `development_command_t`，并在 `AlgorithmBase` 内固定 **50Hz 推理 + 500Hz 指令刷新** 的双线程节奏；示例 `dance_algorithm` 演示 ONNX + `.npz` 动作序列混合管线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ONNX | Open Neural Network Exchange | 跨框架神经网络模型交换格式 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| LCM | Lightweight Communications and Marshalling | 状态订阅与控制命令发布的通信接口 |

## 核心机制（工程切片）

- **观测/动作契约**：子类必须实现 `compute_observation(state)` 与 `process_action(state, action)`；`process_action` 内用 `latest_command_lock` 更新 `latest_command` 字典以适配下游关节分组。
- **模型元数据**：可选从 ONNX metadata 读取 `num_actions`、`default_joint_pos`、`joint_stiffness` 等，减少 YAML 重复配置。
- **RL 模式标志**：`rl_mode.is_rl_mode` 控制是否由状态机侧承担踝部扭矩等细分逻辑（README 语义摘要）。

## 流程总览

```mermaid
flowchart TD
  lcm_in["LCM 订阅<br/>E3_development_state"]
  obs_th["策略线程 ~50Hz<br/>compute_observation"]
  inf["ONNX / PyTorch<br/>前向推理"]
  act["process_action<br/>组装 joint_* 字典"]
  send_th["发令线程 500Hz<br/>latest_command_lock"]
  lcm_out["LCM 发布<br/>E3_development_command"]

  lcm_in --> obs_th --> inf --> act --> send_th --> lcm_out
  lcm_in -.->|"状态缓存"| obs_th
```

## 源码运行时序图

基于 `JackHan-Sdu/Algorithm_Template_For_Developer`（`main` 分支）的 `algorithm_base.py`、`lcm_interface.py` 与示例 `dance_algorithm/run_algorithm.py`、`dance_algorithm/mujoco_simulator.py`：`DanceAlgorithm` 继承 `AlgorithmBase`，初始化期加载 ONNX/PyTorch 模型与 `.npz` 动作数据集并利用真实 LCM 状态做预热；运行期主线程阻塞处理 LCM 状态接收，另起推理线程（配置 `execution.frequency` 如 50Hz）与发令线程（`lcm_send_frequency` 如 500Hz）通过 `latest_command_lock` 解耦。

```mermaid
sequenceDiagram
  autonumber
  participant Sim as MuJoCoDanceSimulator 或宿主控制器
  participant LCM as LCMInterface
  participant Main as 主线程 AlgorithmBase.run
  participant Infer as 推理线程 _execution_loop
  participant Child as DanceAlgorithm 子类
  participant Send as 发令线程 _lcm_send_loop

  Main->>LCM: 订阅 state_channel 并加载 ONNX 或 PT 模型
  Main->>Child: _load_dataset 加载 E3Getup350.npz 并 _warmup_policy
  par 主线程接收状态
    Sim->>LCM: 发布 development_state_t 到 E3_development_state
    Main->>LCM: lcm_interface.handle 触发 _on_state_received
    LCM-->>Main: auto_start 时激活 _start_development_mode
  and 策略推理线程
    Infer->>LCM: get_latest_state 获取最新状态
    Infer->>Child: compute_observation 组装观测与关节顺序映射
    Infer->>Infer: _run_inference 前向推理输出 action
    Infer->>Child: process_action 映射回 XML 顺序并加锁写 latest_command
  and 高频发令线程
    Send->>Send: latest_command_lock 读取最新命令
    Send->>LCM: send_command 发布 development_command_t 到 E3_development_command
    LCM-->>Sim: _handle_development_command 更新关节目标与模式标志
  end
```

- **最短复现路径：** 先启动 `python dance_algorithm/mujoco_simulator.py`，再在另一终端启动 `python dance_algorithm/run_algorithm.py --config config.yaml`（须确保 `config.yaml` 中 `state_channel`、`command_channel`、`robot_id` 与宿主一致）。
- **已核查仓库文件（`JackHan-Sdu/Algorithm_Template_For_Developer` @ `main`）：** `README.md`、`algorithm_base.py`、`lcm_interface.py`、`dance_algorithm/run_algorithm.py`、`dance_algorithm/mujoco_simulator.py`、`lcm-types/development_state_t.lcm`、`lcm-types/development_command_t.lcm`。

## 常见误区或局限

- **2026-11-13 维护复核：** [官方 README（6b7243c）](https://github.com/JackHan-Sdu/Algorithm_Template_For_Developer/blob/6b7243c43823718a834dc4b1db79c59366242d8d/README.md) 将接收主线程、可配置策略推理线程（示例 50 Hz）与 500 Hz 发令线程分开；50 Hz 是推荐配置，并非所有算法固定频率。通道、robot_id 与关节顺序须匹配控制器。本次未运行模板。
- **通道与 robot_id**：`config.yaml` 中 `state_channel` / `command_channel` / `robot_id` 必须与状态机一致，否则表现为「无状态」或指令被丢弃。
- **关节顺序**：README 给出 21 维标准顺序；若模型导出顺序不同，必须在观测与动作两侧同时做映射。

## 与其他页面的关系

- **[WalkE3 控制器](./jackhan-walke3-controller.md)**：模板所对接的宿主进程与模式切换语义。
- **[强化学习](../methods/reinforcement-learning.md)**：策略侧常见输出为关节目标或残差，本模板聚焦部署壳而非训练。

## 参考来源

- [Algorithm_Template_For_Developer 仓库归档](../../sources/repos/jackhan-algorithm-template-for-developer.md)

## 关联页面

- [WalkE3-Controller（人形 RL 部署与 FSM）](./jackhan-walke3-controller.md)
- [JackHan-Sdu WalkE3 / HumanoidE3 工具链生态](./jackhan-walke3-e3-ecosystem.md)
- [强化学习](../methods/reinforcement-learning.md)

## 推荐继续阅读

- 上游仓库 README：<https://github.com/JackHan-Sdu/Algorithm_Template_For_Developer>
